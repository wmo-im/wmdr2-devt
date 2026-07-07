# WMDR2 v0.3.1 strict validFrom E2E fixture fix 3

Install over the previous v0.3.1 strict-validFrom overlay:

```bash
rsync -av /path/to/wmdr2-v031-strict-validfrom-e2e-fix3/ ./
pytest
```

This update only changes the E2E fixture classification. The schema remains strict and the converter still does not invent `validFrom` dates.
