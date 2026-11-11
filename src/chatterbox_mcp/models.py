import json
import ssl

REPO = "ResembleAI/chatterbox"
REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
WEIGHTS = {
    "ve.safetensors": "f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c",
    "t3_cfg.safetensors": "914cb1696f47527fe8852ca8f1fe1fa63cb34f76f9c715e84e067b744dd0da81",
    "s3gen.safetensors": "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e",
    "conds.pt": "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e",
}
FILES = [*WEIGHTS, "tokenizer.json"]


def setup_models(settings):
    import httpx
    import truststore
    from huggingface_hub import hf_hub_download, set_client_factory

    from .download import file_sha256, verified_download

    set_client_factory(
        lambda: httpx.Client(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            timeout=httpx.Timeout(60, read=300),
            follow_redirects=True,
        )
    )
    hashes = {}
    for name in FILES:
        target = settings.checkpoint_dir / name
        if name in WEIGHTS:
            verified_download(REPO, name, REVISION, target)
            if file_sha256(target) != WEIGHTS[name]:
                raise RuntimeError(f"Pinned model verification failed: {name}")
        else:
            hf_hub_download(
                REPO, name, revision=REVISION, local_dir=settings.checkpoint_dir
            )
        hashes[name] = file_sha256(target)
    manifest = {"repo": REPO, "revision": REVISION, "sha256": hashes}
    settings.models.mkdir(parents=True, exist_ok=True)
    (settings.models / "manifest.json").write_text(
        json.dumps(manifest, indent=2), "utf-8"
    )
    return manifest
