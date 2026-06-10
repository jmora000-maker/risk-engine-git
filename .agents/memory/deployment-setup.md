---
name: Deployment setup
description: What is required for this pnpm monorepo to publish successfully to Cloud Run (autoscale).
---

The deployment for this project (risk-engine-jmora0001.replit.app) needs two things to pass the publish pre-check:

1. `.replit` must have a `[deployment]` section with BOTH `deploymentTarget` and a `run` key. The section with only `deploymentTarget` (no `run`) causes "Could not find run command". The section absent entirely causes ".replit is missing the deployment section". The run value is ignored in artifact mode — any valid command satisfies the validator.

2. An artifact must serve at `previewPath = "/"` or the deployed URL returns 404 "no previewable artifacts". The Streamlit app fills this role as `artifacts/streamlit-app/`.

**Why:** The publish pre-check validates .replit synchronously before reading artifact.toml. Artifact mode detection only kicks in at build time, not at the pre-check stage.

**How to apply:** If the [deployment] section ever gets wiped or the run key goes missing, have the user manually restore it. It cannot be written by agent tools (platform blocks all .replit writes). The current value in .replit:
```toml
[deployment]
deploymentTarget = "cloudrun"
run = ["sh", "-c", "cd scripts/src && streamlit run app.py --server.port 8080 --server.address 0.0.0.0"]
```

**verifyAndReplaceArtifactToml** can only replace existing artifact.toml files — it cannot create new ones. For a new artifact, write the temp file then use `cp` in bash to place it.
