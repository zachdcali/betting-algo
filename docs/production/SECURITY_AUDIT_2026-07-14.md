# Runtime And Dependency Security Audit — 2026-07-14

This note separates the active hourly production runtime from the obsolete
Google Cloud UTR deployment path. It records verified findings, changes that
are safe to ship without changing model behavior, and work that remains gated
on model-output parity or authenticated infrastructure access.

## Active hourly runtime

The GitHub Actions hourly pipeline installs the root `requirements.txt`. A
fresh resolution contains two direct packages with known advisories:

- `lxml==6.0.2` is affected by CVE-2026-41066 and is patched in 6.1.0. The
  production parser uses BeautifulSoup over untrusted result HTML, although it
  does not call the vulnerable XML parser defaults. This change moves the pin
  to 6.1.1; ATP parser and settlement fallback tests pass in an isolated 6.1.1
  environment.
- `torch==2.7.1` has current advisories. The most important checkpoint-loading
  issue is fixed in 2.10.0; the smallest candidate with no current OSV finding
  during this audit is 2.12.1. Live artifacts come from the repository release
  and promoted artifacts are checksum-validated, so there is no public upload
  surface. Do not change the live torch version until a checksum-pinned replay
  proves probability parity for every promoted model and all shadow NN models
  receive equivalent checksum coverage.

All state-dictionary loads now request `weights_only=True` explicitly. This is
defense in depth, not a substitute for the gated torch upgrade.

## Model release supply chain

The model cache-miss path previously extracted `models-v1.tar.gz` directly
into the checkout before validating the individual promoted artifacts. A
compromised release archive could therefore overwrite application code before
the registry checks ran.

The release asset is now pinned to archive SHA-256
`a828618d2f5623b81a9868d9bdc927ba875c587fe79aa2aaac19cf913e2bb6a5`.
Extraction rejects absolute paths, traversal, backslashes, members outside
`results/professional_tennis`, links/devices, duplicates, and existing-file
collisions. It stages into a temporary directory before copying validated
model files. The existing 132-file release passes this contract and the
promoted registry validates afterward.

GitHub Actions dependencies are pinned to immutable commit SHAs. Dependabot is
configured for root Python and GitHub Actions updates. Enabling repository
security updates, secret scanning, and push protection remains a GitHub
settings operation.

## Legacy Google Cloud path

`cloud/` is the old Flask/UTR App Engine and Cloud Run path, not the active
TA-based hourly pipeline. Its manifest contains the packages behind the
high/moderate GitHub dependency banner: Gunicorn 21.2.0, Flask 2.3.2, Requests
2.31.0, lxml 4.9.3, and tqdm 4.65.0. The source also imports a missing scraper
module, while the deployment script exposes an unauthenticated `/scrape`
endpoint and injects UTR credentials into revision environment variables.

The predictable App Engine hostname still resolves publicly but returned 503
during this audit. That does not prove whether a Cloud Run revision or App
Engine service remains deployed. Local `gcloud` credentials are expired and
the Cloud Console requires interactive organization SSO, so authenticated
inventory is a stop gate.

After an authorized login, inspect both surfaces:

```bash
gcloud run services describe utr-scraper \
  --project tennis-utr-scraper --region us-central1
gcloud app services list --project tennis-utr-scraper
```

If they are unused, retire the services and then remove the legacy deployment
manifest. If retained, make the service private and redesign authentication,
secrets, startup, and dependencies before redeployment. Do not modernize this
dead path by quietly treating it as the production betting pipeline.

## Remaining gates

1. Run the historical replay under torch 2.7.1 and 2.12.1 and compare every
   promoted probability exactly or within an explicitly reviewed tolerance.
2. Add checksum metadata for all shadow NN artifacts before loading them in the
   hourly runner.
3. Produce a fully resolved, hashed Python 3.12 Linux production lock and run
   `pip check` plus `pip-audit --strict` in CI.
4. Inventory and retire or secure the legacy Google Cloud services after SSO.
5. Enable GitHub security updates, secret scanning, and push protection.
