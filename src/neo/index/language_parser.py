"""Language-agnostic code parsing using tree-sitter.

Provides unified parsing across multiple programming languages for semantic
code chunking. Replaces Python-specific AST parsing with tree-sitter.

Supported languages:
- Python, C#, TypeScript, JavaScript, Java, Go, Rust, C/C++
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import hashlib
import time

# Deprecated re-export of the canonical extension → tree-sitter
# language map. New code should `from neo.languages import
# EXTENSION_TO_LANGUAGE` (or `language_for_path`). This shim exists
# only so existing imports of `LANGUAGE_MAP` continue to work; it
# will be removed once all consumers have migrated.
from neo.languages import EXTENSION_TO_LANGUAGE as LANGUAGE_MAP  # noqa: F401

from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_parser, get_language

# tree-sitter-language-pack uses `csharp` while the rest of this
# codebase uses the tree-sitter canonical name `c_sharp` (matching
# what node types and grammar names look like in the parser).
# Translate at the parser-load boundary so the canonical name flows
# through QUERIES / EDGE_QUERIES / EXTENSION_TO_LANGUAGE unchanged.
_PARSER_NAME_ALIASES = {"c_sharp": "csharp"}


def _resolve_parser_name(language: str) -> str:
    return _PARSER_NAME_ALIASES.get(language, language)

# Kept for backward compatibility — older consumers conditionally gated
# on TREE_SITTER_AVAILABLE. tree-sitter is now a hard dependency, so
# this is always True. Remove once all references are gone.
TREE_SITTER_AVAILABLE = True

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_LENGTH = 2000  # Characters per chunk


@dataclass
class CodeEdge:
    """A relationship between code symbols (import, inheritance, etc.)."""
    source_file: str
    source_symbol: str  # Symbol where the edge originates (or "" for file-level)
    target_symbol: str  # Symbol being referenced
    edge_type: str      # "imports", "inherits", "implements"
    line_number: int


@dataclass
class CodeChunk:
    """A semantic chunk of code (function, class, etc.)."""
    file_path: str
    chunk_id: str
    content: str
    chunk_type: str
    start_line: int
    end_line: int
    symbols: List[str]
    imports: List[str]
    file_hash: str
    indexed_at: float


# Tree-sitter queries for extracting code structures
# Format: language -> (functions_query, classes_query)
QUERIES = {
    'python': {
        'functions': """
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @params
                body: (block) @body) @function
        """,
        'classes': """
            (class_definition
                name: (identifier) @name
                body: (block) @body) @class
        """,
        'module_doc': """
            (module
                . (expression_statement
                    (string) @doc))
        """
    },
    'c_sharp': {
        'functions': """
            (method_declaration
                name: (identifier) @name
                parameters: (parameter_list) @params
                body: (_) @body) @method
        """,
        'classes': """
            (class_declaration
                name: (identifier) @name
                body: (declaration_list) @body) @class
        """,
        'interfaces': """
            (interface_declaration
                name: (identifier) @name
                body: (declaration_list) @body) @interface
        """
    },
    'typescript': {
        'functions': """
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body) @function
        """,
        'classes': """
            (class_declaration
                name: (type_identifier) @name
                body: (class_body) @body) @class
        """,
        'interfaces': """
            (interface_declaration
                name: (type_identifier) @name
                body: (object_type) @body) @interface
        """
    },
    'javascript': {
        'functions': """
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body) @function
        """,
        'classes': """
            (class_declaration
                name: (identifier) @name
                body: (class_body) @body) @class
        """
    },
    'java': {
        'functions': """
            (method_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (block) @body) @method
        """,
        'classes': """
            (class_declaration
                name: (identifier) @name
                body: (class_body) @body) @class
        """
    },
    'go': {
        'functions': """
            (function_declaration
                name: (identifier) @name
                parameters: (parameter_list) @params
                body: (block) @body) @function
        """,
        'types': """
            (type_declaration
                (type_spec
                    name: (type_identifier) @name
                    type: (struct_type) @body)) @struct
        """
    },
    'rust': {
        'functions': """
            (function_item
                name: (identifier) @name
                parameters: (parameters) @params
                body: (block) @body) @function
        """,
        'types': """
            (struct_item
                name: (type_identifier) @name
                body: (field_declaration_list) @body) @struct
        """
    },
    'c': {
        'functions': """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name
                    parameters: (parameter_list) @params)
                body: (compound_statement) @body) @function
        """,
        'structs': """
            (struct_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body) @struct
        """
    },
    'cpp': {
        'functions': """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name
                    parameters: (parameter_list) @params)
                body: (compound_statement) @body) @function
        """,
        'classes': """
            (class_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body) @class
        """
    },
}

# TSX uses same queries as TypeScript
QUERIES['tsx'] = QUERIES['typescript']

# Edge queries for relationship extraction (imports, inheritance)
EDGE_QUERIES = {
    'python': {
        'imports': """
            (import_statement
                name: (dotted_name) @module) @import
        """,
        'import_from': """
            (import_from_statement
                module_name: (dotted_name) @module
                name: (dotted_name) @name) @import
        """,
        'inheritance': """
            (class_definition
                name: (identifier) @class_name
                superclasses: (argument_list
                    (identifier) @base)) @class_def
        """,
    },
    'c_sharp': {
        'imports': """
            (using_directive
                (qualified_name) @module) @import
        """,
        'inheritance': """
            (class_declaration
                name: (identifier) @class_name
                bases: (base_list
                    (identifier) @base)) @class_def
        """,
    },
    'typescript': {
        'imports': """
            (import_statement
                source: (string) @module) @import
        """,
        'inheritance': """
            (class_declaration
                name: (type_identifier) @class_name
                (class_heritage
                    (extends_clause
                        value: (identifier) @base))) @class_def
        """,
        'implements': """
            (class_declaration
                name: (type_identifier) @class_name
                (class_heritage
                    (implements_clause
                        (type_identifier) @interface))) @class_def
        """,
    },
    'javascript': {
        'imports': """
            (import_statement
                source: (string) @module) @import
        """,
        'inheritance': """
            (class_declaration
                name: (identifier) @class_name
                (class_heritage
                    (identifier) @base)) @class_def
        """,
    },
    'java': {
        'imports': """
            (import_declaration
                (scoped_identifier) @module) @import
        """,
        'inheritance': """
            (class_declaration
                name: (identifier) @class_name
                (superclass
                    (type_identifier) @base)) @class_def
        """,
        'implements': """
            (class_declaration
                name: (identifier) @class_name
                (super_interfaces
                    (type_list
                        (type_identifier) @interface))) @class_def
        """,
    },
    'go': {
        'imports': """
            (import_spec
                path: (interpreted_string_literal) @module) @import
        """,
    },
    'rust': {
        'imports': """
            (use_declaration
                argument: (scoped_identifier) @module) @import
        """,
    },
    'c': {
        'imports': """
            (preproc_include
                path: (_) @module) @import
        """,
    },
    'cpp': {
        'imports': """
            (preproc_include
                path: (_) @module) @import
        """,
        'inheritance': """
            (class_specifier
                name: (type_identifier) @class_name
                (base_class_clause
                    (type_identifier) @base)) @class_def
        """,
    },
}

EDGE_QUERIES['tsx'] = EDGE_QUERIES['typescript']


class TreeSitterParser:
    """
    Multi-language code parser using tree-sitter.

    Extracts semantic chunks (functions, classes, methods) from source files
    across multiple programming languages.
    """

    def __init__(self):
        """Initialize parser with lazy loading."""
        self.parsers: Dict[str, Any] = {}  # Lazy-loaded parsers per language
        self.languages: Dict[str, Any] = {}  # Language objects
        self.compiled_queries: Dict[str, Dict[str, Any]] = {}  # Compiled queries

    def supports_extension(self, ext: str) -> bool:
        """Check if file extension is supported."""
        return ext.lower() in LANGUAGE_MAP

    def detect_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file extension."""
        ext = file_path.suffix.lower()
        return LANGUAGE_MAP.get(ext)

    def _get_parser(self, language: str):
        """Get or create parser for language (lazy loading)."""
        if language not in self.parsers:
            try:
                parser_name = _resolve_parser_name(language)
                self.parsers[language] = get_parser(parser_name)
                self.languages[language] = get_language(parser_name)
                logger.debug(f"Loaded tree-sitter parser for {language}")
            except Exception as e:
                logger.error(f"Failed to load parser for {language}: {e}")
                raise
        return self.parsers[language]

    def _get_query(self, language: str, query_name: str):
        """Get or compile query for language."""
        cache_key = f"{language}:{query_name}"
        if cache_key not in self.compiled_queries:
            if language not in QUERIES or query_name not in QUERIES[language]:
                return None

            query_text = QUERIES[language][query_name]
            lang_obj = self.languages.get(language)
            if not lang_obj:
                # Ensure language is loaded
                self._get_parser(language)
                lang_obj = self.languages[language]

            try:
                # tree-sitter 0.23+: Query(language, text). Older
                # Language.query(text) is deprecated and removed in
                # newer versions.
                self.compiled_queries[cache_key] = Query(lang_obj, query_text)
                logger.debug(f"Compiled query {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to compile query {cache_key}: {e}")
                return None

        return self.compiled_queries.get(cache_key)

    @staticmethod
    def _run_query(query, root_node) -> List[Tuple[Any, str]]:
        """Run a query and return captures as the legacy list of
        (node, capture_name) tuples in document order.

        tree-sitter 0.23+ returns `dict[capture_name, list[Node]]`
        from QueryCursor.captures(); the existing _process_*_captures
        methods expect the old list-of-tuples shape, and
        _process_inheritance_captures specifically depends on document
        order (it tracks `current_class` from a `class_name` capture
        and applies it to subsequent `base`/`interface` captures).
        """
        cursor = QueryCursor(query)
        captures_dict = cursor.captures(root_node)
        pairs: List[Tuple[Any, str]] = []
        for capture_name, nodes in captures_dict.items():
            for node in nodes:
                pairs.append((node, capture_name))
        pairs.sort(key=lambda pair: pair[0].start_byte)
        return pairs

    def parse_file(
        self,
        file_path: Path,
        content: str,
        language: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Extract semantic chunks from file.

        Args:
            file_path: Path to source file
            content: File content as string
            language: Language name (auto-detected if None)

        Returns:
            List of CodeChunk objects
        """
        if language is None:
            language = self.detect_language(file_path)

        if not language:
            logger.warning(f"Unknown language for {file_path}")
            return []

        if language not in QUERIES:
            logger.warning(f"No queries defined for {language}")
            return []

        try:
            # Parse file
            parser = self._get_parser(language)
            tree = parser.parse(bytes(content, 'utf8'))

            # Extract chunks
            chunks = []
            file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Extract edges for import list
            edges = self._extract_edges(tree, content, str(file_path), language)
            import_symbols = [e.target_symbol for e in edges if e.edge_type == "imports"]

            # Extract each type of construct (functions, classes, etc.)
            for query_name in QUERIES[language].keys():
                query = self._get_query(language, query_name)
                if not query:
                    continue

                captures = self._run_query(query, tree.root_node)
                chunks.extend(self._process_captures(
                    captures,
                    content,
                    str(file_path),
                    file_hash,
                    query_name,
                    import_symbols
                ))

            return chunks

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []

    def extract_edges(
        self,
        file_path: Path,
        content: str,
        language: Optional[str] = None
    ) -> List[CodeEdge]:
        """
        Extract relationship edges (imports, inheritance) from file.

        Args:
            file_path: Path to source file
            content: File content as string
            language: Language name (auto-detected if None)

        Returns:
            List of CodeEdge objects
        """
        if language is None:
            language = self.detect_language(file_path)

        if not language:
            return []

        try:
            parser = self._get_parser(language)
            tree = parser.parse(bytes(content, 'utf8'))
            return self._extract_edges(tree, content, str(file_path), language)
        except Exception as e:
            logger.error(f"Failed to extract edges from {file_path}: {e}")
            return []

    def _extract_edges(
        self,
        tree,
        content: str,
        file_path: str,
        language: str
    ) -> List[CodeEdge]:
        """Extract edges from a parsed tree."""
        if language not in EDGE_QUERIES:
            return []

        edges = []
        content_bytes = content.encode('utf8')

        for query_name, query_text in EDGE_QUERIES[language].items():
            try:
                lang_obj = self.languages.get(language)
                if not lang_obj:
                    self._get_parser(language)
                    lang_obj = self.languages[language]

                query = Query(lang_obj, query_text)
                captures = self._run_query(query, tree.root_node)
            except Exception as e:
                logger.debug(f"Edge query {query_name} failed for {language}: {e}")
                continue

            if query_name in ('imports', 'import_from'):
                edges.extend(self._process_import_captures(
                    captures, content_bytes, file_path
                ))
            elif query_name == 'inheritance':
                edges.extend(self._process_inheritance_captures(
                    captures, content_bytes, file_path, "inherits"
                ))
            elif query_name == 'implements':
                edges.extend(self._process_inheritance_captures(
                    captures, content_bytes, file_path, "implements"
                ))

        return edges

    def _process_import_captures(
        self,
        captures: List[Tuple],
        content_bytes: bytes,
        file_path: str
    ) -> List[CodeEdge]:
        """Process import query captures into CodeEdges."""
        edges = []
        for node, capture_name in captures:
            if capture_name == 'module':
                module_text = content_bytes[node.start_byte:node.end_byte].decode('utf8')
                # Strip quotes from string literals (JS/TS/Go imports)
                module_text = module_text.strip('\'"')
                line = content_bytes[:node.start_byte].count(b'\n') + 1
                edges.append(CodeEdge(
                    source_file=file_path,
                    source_symbol="",
                    target_symbol=module_text,
                    edge_type="imports",
                    line_number=line,
                ))
        return edges

    def _process_inheritance_captures(
        self,
        captures: List[Tuple],
        content_bytes: bytes,
        file_path: str,
        edge_type: str
    ) -> List[CodeEdge]:
        """Process inheritance/implements query captures into CodeEdges."""
        edges = []
        # Track class_name and base pairs from captures
        current_class = None
        for node, capture_name in captures:
            text = content_bytes[node.start_byte:node.end_byte].decode('utf8')
            if capture_name == 'class_name':
                current_class = text
            elif capture_name in ('base', 'interface') and current_class:
                line = content_bytes[:node.start_byte].count(b'\n') + 1
                edges.append(CodeEdge(
                    source_file=file_path,
                    source_symbol=current_class,
                    target_symbol=text,
                    edge_type=edge_type,
                    line_number=line,
                ))
        return edges

    def _process_captures(
        self,
        captures: List[Tuple],
        content: str,
        file_path: str,
        file_hash: str,
        query_type: str,
        import_symbols: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """Process tree-sitter query captures into CodeChunks."""
        chunks = []
        content_bytes = content.encode('utf8')

        # Group captures by their parent construct
        constructs = {}  # node_id -> {name, body, start, end}

        for node, capture_name in captures:
            # Find the parent construct (the one with full definition)
            if capture_name in ['function', 'class', 'method', 'interface', 'struct']:
                node_id = id(node)
                if node_id not in constructs:
                    constructs[node_id] = {
                        'node': node,
                        'type': capture_name,
                        'name': None,
                        'symbols': []
                    }
            elif capture_name == 'name':
                # Find which construct this name belongs to
                name_text = content_bytes[node.start_byte:node.end_byte].decode('utf8')

                # Find the smallest construct that contains this name
                # (handles nested constructs correctly)
                containing_construct = None
                smallest_size = float('inf')

                for construct_id, construct in constructs.items():
                    construct_node = construct['node']
                    # Check if name is within this construct
                    if (node.start_byte >= construct_node.start_byte and
                        node.end_byte <= construct_node.end_byte):
                        # Track the smallest containing construct
                        construct_size = construct_node.end_byte - construct_node.start_byte
                        if construct_size < smallest_size:
                            smallest_size = construct_size
                            containing_construct = construct

                if containing_construct is not None:
                    if containing_construct['name'] is None:
                        containing_construct['name'] = name_text
                        containing_construct['symbols'].append(name_text)
                    else:
                        # Additional symbol (e.g., method in a class)
                        if name_text not in containing_construct['symbols']:
                            containing_construct['symbols'].append(name_text)

        # Create chunks from constructs
        for construct_id, construct in constructs.items():
            node = construct['node']
            name = construct.get('name', 'anonymous')

            # Extract content
            chunk_content = content_bytes[node.start_byte:node.end_byte].decode('utf8')

            # Truncate if too long
            if len(chunk_content) > MAX_CHUNK_LENGTH:
                chunk_content = chunk_content[:MAX_CHUNK_LENGTH]

            # Calculate line numbers
            start_line = content[:node.start_byte].count('\n') + 1
            end_line = content[:node.end_byte].count('\n') + 1

            # Create chunk
            chunk = CodeChunk(
                file_path=file_path,
                chunk_id=f"{construct['type']}:{name}",
                content=chunk_content,
                chunk_type=construct['type'],
                start_line=start_line,
                end_line=end_line,
                symbols=construct.get('symbols', []),
                imports=import_symbols or [],
                file_hash=file_hash,
                indexed_at=time.time()
            )
            chunks.append(chunk)

        return chunks

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(set(LANGUAGE_MAP.values()))

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(LANGUAGE_MAP.keys())
