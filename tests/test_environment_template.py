""".env.example is checked against the code, not maintained beside it.

A configuration template is only useful if it is complete, and a hand-maintained
one stops being complete on the first commit that adds a variable and forgets
the docs. From then on it is worse than nothing: an operator who reads it
believes they have seen the whole surface.

So this file discovers the surface from the source and compares. It reads
nothing at runtime and imports neither package -- the template spans the
quantitative core, the API and the collector, which have three different
dependency sets and run under two different CI jobs, and a documentation check
that could only run in one of them would be exactly the kind of gap it exists
to close.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / ".env.example"
PATHS_MODULE = REPO_ROOT / "market_intelligence" / "ops" / "paths.py"

#: Where runtime configuration is read. `research/` is excluded because it holds
#: superseded implementations, and `tests/` because a test setting a variable is
#: not the application reading one.
SOURCE_ROOTS = (
    REPO_ROOT / "btc_forecaster",
    REPO_ROOT / "market_intelligence",
)
SOURCE_FILES = (
    REPO_ROOT / "api_server.py",
    REPO_ROOT / "btc-intel.py",
    REPO_ROOT / "btc-intel-api.py",
)

#: `btc_forecaster/config/settings.py` reads the environment through five typed
#: helpers rather than calling os.getenv directly, so the variable names live at
#: the call sites. Named here so those twenty-one names are discovered rather
#: than quietly missed.
ENV_HELPERS = frozenset({"_env_str", "_env_int", "_env_float", "_env_bool", "_env_tuple"})

#: Documented in `.env.example` under the AWS note, read by botocore rather than
#: by this repository. Listing them as discovered would be wrong -- nothing here
#: reads them -- and leaving them undocumented would be worse, because S3 sync
#: does not work without them.
EXTERNALLY_READ = frozenset({"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"})


def python_sources() -> list[Path]:
    files = [path for path in SOURCE_FILES if path.exists()]
    for root in SOURCE_ROOTS:
        files.extend(sorted(root.rglob("*.py")))
    return files


def _literal(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _called_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        # `.get` alone is far too broad -- every dictionary lookup in the
        # repository is one. Only a lookup whose receiver is the environment
        # counts, which is why the receiver is checked rather than the method.
        if func.attr == "get":
            return "environ.get" if _is_environ(func.value) else ""
        return func.attr
    return ""


def _is_environ(node: ast.AST) -> bool:
    """True for `os.environ` and for a bare `environ` imported from os."""
    if isinstance(node, ast.Attribute):
        return node.attr == "environ"
    return isinstance(node, ast.Name) and node.id == "environ"


def collector_path_variables() -> list[str]:
    """The names StoragePaths composes from ENV_PREFIX.

    Composed at runtime from an f-string, so no literal scan can see them. The
    module declares them as constants for exactly this reason.
    """
    tree = ast.parse(PATHS_MODULE.read_text(encoding="utf-8"))
    prefix = ""
    suffixes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        name = next((t.id for t in targets if isinstance(t, ast.Name)), None)
        if name == "ENV_PREFIX":
            prefix = _literal(node.value) or ""
        elif name == "PATH_ENV_SUFFIXES" and isinstance(node.value, ast.Tuple):
            suffixes = [text for element in node.value.elts if (text := _literal(element))]
    assert prefix and suffixes, "paths.py must declare ENV_PREFIX and PATH_ENV_SUFFIXES"
    return [f"{prefix}{suffix}" for suffix in suffixes]


def discovered_variables() -> dict[str, list[str]]:
    """Environment variable -> the files that read it, found statically."""
    found: dict[str, list[str]] = {}

    def record(name: str, path: Path) -> None:
        found.setdefault(name, []).append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))

    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # A module-level `SOMETHING_ENV = "NAME"` declares a read that no scan
        # of `os.environ` can find, because the lookup happens through an
        # aliased mapping. `CONTACT_ENV` is the case that matters: it is the one
        # variable a collection deployment genuinely requires, and it is read
        # via `env.get(...)` where `env` may be a test double.
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            name = next((t.id for t in targets if isinstance(t, ast.Name)), None)
            value = _literal(node.value) if node.value is not None else None
            if name and name.endswith("_ENV") and value and value.isupper():
                record(value, path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _called_name(node)
                # os.getenv("X") / os.environ.get("X") / _env_str("X", ...)
                if (name in {"getenv", "environ.get"} or name in ENV_HELPERS) and node.args:
                    literal = _literal(node.args[0])
                    if literal and literal.isupper() and literal.replace("_", "").isalnum():
                        record(literal, path)
                # ProviderDeclaration(credentials_env="X")
                for keyword in node.keywords:
                    if keyword.arg == "credentials_env":
                        literal = _literal(keyword.value)
                        if literal:
                            record(literal, path)
            # os.environ["X"]
            if isinstance(node, ast.Subscript) and _is_environ(node.value):
                literal = _literal(node.slice)
                if literal:
                    record(literal, path)

    for variable in collector_path_variables():
        record(variable, PATHS_MODULE)
    return found


#: BTC_INTEL_CONTACT=you@example.org, and the commented-out form beside it.
_ASSIGNMENT = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$")


def documented() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT.match(line)
        if match:
            entries[match.group(1)] = match.group(2).strip()
    return entries


@pytest.fixture(scope="module")
def template() -> dict[str, str]:
    return documented()


@pytest.fixture(scope="module")
def in_code() -> dict[str, list[str]]:
    return discovered_variables()


class TestTheTemplateExists:
    def test_it_is_committed(self) -> None:
        assert TEMPLATE.exists()

    def test_dot_env_itself_is_ignored(self) -> None:
        """The template is committed; the filled-in copy never is."""
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in [line.strip() for line in ignore]

    def test_it_names_its_four_categories(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        for category in ("REQUIRED", "OPTIONAL", "DEPLOYMENT-ONLY", "RESEARCH-ONLY"):
            assert category in text, category

    def test_it_says_nothing_loads_it_automatically(self) -> None:
        """There is no dotenv loader in this repository, deliberately. A
        template that implies one sends an operator looking for a bug in the
        code when the variable they set was simply never exported."""
        text = TEMPLATE.read_text(encoding="utf-8")
        assert "reads `.env` automatically" in text


class TestTheTemplateCoversTheCode:
    def test_every_variable_the_code_reads_is_documented(
        self, template: dict[str, str], in_code: dict[str, list[str]]
    ) -> None:
        missing = {
            name: sorted(set(files)) for name, files in in_code.items() if name not in template
        }
        assert not missing, f"read by the code, absent from .env.example: {missing}"

    def test_it_documents_nothing_the_code_does_not_read(
        self, template: dict[str, str], in_code: dict[str, list[str]]
    ) -> None:
        """Decoration is the other failure mode. A variable listed here that
        nothing reads is an instruction that silently does nothing."""
        extra = set(template) - set(in_code) - EXTERNALLY_READ
        assert not extra, f"documented but never read: {sorted(extra)}"

    def test_the_collector_surface_is_complete(self) -> None:
        """PATH_ENV_SUFFIXES is what the template is checked against, so it has
        to match what _override is actually called with."""
        source = PATHS_MODULE.read_text(encoding="utf-8")
        called = set(re.findall(r'_override\(\s*"([A-Z_]+)"', source))
        declared = {name.removeprefix("BTC_INTEL_") for name in collector_path_variables()}
        assert called <= declared, f"overridden but undeclared: {sorted(called - declared)}"
        assert declared - called == {"STATE_ROOT"}, sorted(declared - called)

    def test_the_variables_that_matter_are_present(self, template: dict[str, str]) -> None:
        for name in (
            "BTC_INTEL_CONTACT",
            "BTC_INTEL_STATE_ROOT",
            "API_TOKEN",
            "OUTPUT_DIR",
            "WF_FOLDS",
        ):
            assert name in template, name


class TestTheDeploymentTemplateAgreesWithIt:
    """`deploy/collector.env.example` is not a duplicate of `.env.example`.

    It is the file `deploy/install.sh` copies to `/etc/btc-intel/collector.env`
    and systemd reads as its `EnvironmentFile`, so it holds the collector subset
    and the host-specific guidance that goes with it. What it must never do is
    name a variable the repository-wide template does not, because then the
    complete template is not complete.
    """

    COLLECTOR_TEMPLATE = REPO_ROOT / "deploy" / "collector.env.example"

    def test_it_is_still_the_file_the_installer_copies(self) -> None:
        installer = (REPO_ROOT / "deploy" / "install.sh").read_text(encoding="utf-8")
        assert "deploy/collector.env.example" in installer

    def test_it_names_nothing_the_repository_template_omits(
        self, template: dict[str, str]
    ) -> None:
        collector: set[str] = set()
        for line in self.COLLECTOR_TEMPLATE.read_text(encoding="utf-8").splitlines():
            match = _ASSIGNMENT.match(line)
            if match:
                collector.add(match.group(1))
        assert collector <= set(template), sorted(collector - set(template))

    def test_it_carries_no_address_either(self) -> None:
        text = self.COLLECTOR_TEMPLATE.read_text(encoding="utf-8").casefold()
        for fragment in TestTheTemplateCarriesNoSecrets.FORBIDDEN:
            assert fragment not in text, fragment


class TestTheTemplateCarriesNoSecrets:
    #: A dedicated project address was configured into the deployment host in an
    #: earlier track, on the explicit condition that it never enter source
    #: control. This is the check that keeps that true.
    FORBIDDEN = ("santos.cesarndrei", "gmail.com", "ndreisantos")

    def test_no_real_contact_address(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8").casefold()
        for fragment in self.FORBIDDEN:
            assert fragment not in text, fragment

    def test_the_contact_is_a_placeholder(self, template: dict[str, str]) -> None:
        value = template["BTC_INTEL_CONTACT"]
        assert value.endswith("example.org") or not value, value

    def test_no_credential_carries_a_value(self, template: dict[str, str]) -> None:
        """Every credential is present as an empty or obviously fake key. A
        template that ships a working value is a template that ships a secret."""
        for name, value in template.items():
            if any(token in name for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                assert value in {"", "change-me"}, f"{name} carries a value"
