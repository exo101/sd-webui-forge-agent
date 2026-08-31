from __future__ import annotations

from typing import TYPE_CHECKING, Any

from huggingface_hub import PyTorchModelHubMixin, whoami

from .utils import set_doc

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class Revised_Mixin(PyTorchModelHubMixin):
    @set_doc(PyTorchModelHubMixin.push_to_hub.__doc__)
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict | DataclassInstance | None = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        model_card_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if model_card_kwargs is None:
            model_card_kwargs = {}
        if "/" not in repo_id:
            username = whoami()["name"]
            repo_id = f"{username}/{repo_id}"
        model_card_kwargs["repo_id"] = repo_id
        return super().push_to_hub(
            repo_id,
            config=config,
            commit_message=commit_message,
            private=private,
            token=token,
            branch=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            model_card_kwargs=model_card_kwargs,
        )