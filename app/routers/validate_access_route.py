from fastapi import APIRouter, Depends
from app.models.schemas import ValidateAccessRequest, ValidateAccessResponse
from app.core.dependecies import get_dependencies, Dependencies

router = APIRouter()

@router.post("/validate-access", response_model=ValidateAccessResponse)
async def validate_access(
    request: ValidateAccessRequest,
    deps: Dependencies = Depends(get_dependencies),
):
    allowed = deps.acl_checker.can_access(request.user_role, request.document_type)

    reason = None
    if not allowed:
        reason = f"User role '{request.user_role}' cannot access '{request.document_type}' documents"

    redaction_hints = []
    redaction_rules = deps.acl_checker.get_redaction_rules(request.user_role)
    if redaction_rules:
        redaction_hints = [rule.get("pattern", "") for rule in redaction_rules]

    return ValidateAccessResponse(
        allowed=allowed,
        reason=reason,
        redaction_hints=redaction_hints,
    )