# Organization-bound Lattice nodes

An organization node keeps the issuer, account, and organization supplied at
launch. Changing CAR's displayed organization does not change an existing node,
its consumer grants, project approvals, or private storage. CAR verifies membership
and admission; a context JSON file alone grants no authority.

Start an organization node from the authenticated operator session with an
explicit context and native-owner configuration:

```sh
neo lattice join --cwd /absolute/repository \
  --work-context-file /absolute/work-context.json \
  --native-owner-config /absolute/native-owner.json
neo lattice status --cwd /absolute/repository \
  --work-context-file /absolute/work-context.json
neo lattice leave --cwd /absolute/repository \
  --work-context-file /absolute/work-context.json
```

The context contains `api_base`, `account_id`, and `organization_id`. Approve
shared repositories explicitly with repeatable `--project-id repo:<64 lowercase
hex>` arguments on the initial join. Status, leave, and repeated join reuse the
original saved native-owner configuration. Supplying a changed configuration or
project scope refuses the operation.

The native-owner JSON contains these exact fields:

| Field | Value |
| --- | --- |
| `guardian_python` | Absolute installed guardian virtual-environment interpreter path; preserve its symlink |
| `guardian_scripts` | Absolute installed guardian scripts directory |
| `child_python` | Absolute isolated Neo interpreter path |
| `child_sdk` | Absolute isolated CAR native SDK file path |
| `child_package` | Absolute isolated Neo package directory |
| `ownership_root` | Absolute private owner directory, with no symlink path components |
| `pins` | Map of absolute artifact paths to SHA256 digests |

Pins must include both interpreters, the guardian entrypoint, child SDK, and every
Python source in the child package. Use the installed guardian inventory to pin
its payload. Neo checks the pins before each operation; the installed guardian
also verifies its dependencies. The child package must have no source symlinks or
`.env` fallback. The guardian and child SDK are separate artifacts and can have
different hashes.

The approved child specification is tokenless, uses `auto_start=false`, and has
an empty environment map. The installed guardian owns launch credentials and
process cleanup. Neo saves uncertain admission and cleanup before effects and
refuses replacement work while either remains pending. A successful leave
requires the original task's termination and cleanup receipts. A subsequent
explicit join rechecks the original daemon, context, task, specification, private
resource, and project approvals before archiving it and admitting a new identity.
Legacy managed enrollments require independently proven original termination and
explicit migration; this bridge does not migrate them automatically.

Scoped consumers use their admitted CAR runtime for routing and inference, with
private fact, episode, log, and metric storage. They run in advisory mode and
reauthorize before and after producing results. Failed admission, revocation,
changed context, or reconnect refusal cannot select a personal provider fallback.

The automated controls exercise admission, isolation, refused context changes,
consumer revocation, reconnects, and original-owner lifecycle gates. Mocked
controls and synthetic installed-SDK fixtures do not establish real multiplayer
acceptance: that still requires connected coding assistants and distinct users
and devices across organizations. Task termination and organization membership
revocation are separate checks.

This integration requires CAR builds implementing the scoped-consumer and native
guardian contracts described above. Neo 0.56.0 ships the consumer integration;
installed native-owner restart recovery and cleanup acceptance, and real
multiplayer acceptance, remain unverified. A version label alone does not prove
that the required contracts or pinned native artifacts are present.
