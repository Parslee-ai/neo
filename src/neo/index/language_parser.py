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

try:
    # Try py-tree-sitter-languages first (newer package)
    try:
        from tree_sitter_languages import get_parser, get_language
    except ImportError:
        # Fall back to individual language packages if available
        # This will fail gracefully if not available
        raise ImportError("tree-sitter-languages not available")
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Create dummy functions for when tree-sitter is not available
    def get_parser(lang):
        raise ImportError(f"tree-sitter not available, cannot parse {lang}")
    def get_language(lang):
        raise ImportError(f"tree-sitter not available, cannot get language {lang}")

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_LENGTH = 2000  # Characters per chunk


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


# Language configuration: maps file extension to tree-sitter language name
LANGUAGE_MAP = {
    '.py': 'python',
    '.pyi': 'python',
    '.cs': 'c_sharp',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.hh': 'cpp',
}

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


class TreeSitterParser:
    """
    Multi-language code parser using tree-sitter.

    Extracts semantic chunks (functions, classes, methods) from source files
    across multiple programming languages.
    """

    def __init__(self):
        """Initialize parser with lazy loading."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter-languages not available. "
                "Install with: pip install tree-sitter-languages"
            )

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
                self.parsers[language] = get_parser(language)
                self.languages[language] = get_language(language)
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
                # In tree-sitter 0.21.x, queries are created via Language.query()
                self.compiled_queries[cache_key] = lang_obj.query(query_text)
                logger.debug(f"Compiled query {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to compile query {cache_key}: {e}")
                return None

        return self.compiled_queries.get(cache_key)

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

            # Extract each type of construct (functions, classes, etc.)
            for query_name in QUERIES[language].keys():
                query = self._get_query(language, query_name)
                if not query:
                    continue

                captures = query.captures(tree.root_node)
                chunks.extend(self._process_captures(
                    captures,
                    content,
                    str(file_path),
                    file_hash,
                    query_name
                ))

            return chunks

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []

    def _process_captures(
        self,
        captures: List[Tuple],
        content: str,
        file_path: str,
        file_hash: str,
        query_type: str
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
                imports=[],  # TODO: Extract imports
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
