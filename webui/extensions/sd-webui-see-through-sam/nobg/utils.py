from functools import wraps


# update docs for specific functions
def set_doc(doc):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = doc
        return wrapper

    return decorator


# general template for models
def model_card_template(
    *, class_name: str, default_repo: str, citation: str | None = None
) -> str:
    citation_section = (
        f"""
## Citation
If you use this model, please cite:
```bibtex
{citation}
```
"""
        if citation
        else ""
    )
    return f"""
---
{{{{ card_template }} }}
---
## Model Details
This is a {{class_name}} model from the [nobg](https://github.com/echo714/nobg) project.

**Default repository:** [{default_repo}](https://huggingface.co/{default_repo})
{citation_section}
"""