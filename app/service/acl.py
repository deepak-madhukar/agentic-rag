import logging
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ACLChecker:
    def __init__(self, acl_rules_file: Path):
        self.rules = self._load_rules(acl_rules_file)

    def _load_rules(self, rules_file: Path) -> dict:
        if not rules_file.exists():
            logger.warning(f"ACL rules file not found: {rules_file}")
            return self._default_rules()

        try:
            if rules_file.suffix == ".json":
                with open(rules_file, "r") as f:
                    return json.load(f)
            else:
                return self._parse_yaml_file(rules_file)
        except Exception as e:
            logger.warning(f"Failed to load ACL rules: {e}; using default")
            return self._default_rules()

    def _parse_yaml_file(self, rules_file: Path) -> dict:
        try:
            import yaml

            with open(rules_file, "r") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed; using JSON fallback or defaults")
            return self._default_rules()

    def _default_rules(self) -> dict:
        return {
            "roles": {
                "Admin": {"can_access": ["*"], "redaction_rules": []},
                "Manager": {"can_access": ["PUBLIC", "INTERNAL", "TEAM"], "redaction_rules": []},
                "Analyst": {"can_access": ["PUBLIC", "INTERNAL"], "redaction_rules": []},
                "Contractor": {"can_access": ["PUBLIC"], "redaction_rules": []},
            }
        }

    def can_access(self, user_role: str, document_type: str) -> bool:
        roles = self.rules.get("roles", {})
        role_config = roles.get(user_role, {})
        allowed = role_config.get("can_access", ["PUBLIC"])

        if "*" in allowed:
            return True

        return document_type in allowed

    def get_redaction_rules(self, user_role: str) -> list[dict]:
        roles = self.rules.get("roles", {})
        role_config = roles.get(user_role, {})
        return role_config.get("redaction_rules", [])

    def redact_content(self, content: str, user_role: str) -> str:
        rules = self.get_redaction_rules(user_role)

        import re

        redacted = content
        for rule in rules:
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "[REDACTED]")
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

        return redacted

    def validate_chunk_access(self, chunk: dict, user_role: str) -> tuple[bool, Optional[str]]:
        doc_type = chunk.get("document_type", "PUBLIC")

        if not self.can_access(user_role, doc_type):
            return False, f"Document type {doc_type} not accessible by {user_role}"

        return True, None