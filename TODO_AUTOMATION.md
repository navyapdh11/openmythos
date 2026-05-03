# OpenMythos Automation Manifest (TODO)

This manifest outlines the pending tasks to be executed upon environment initiation to complete the `OpenMythos` infrastructure upgrade.

## 1. Environment Preparation
- [ ] **Infrastructure Expansion**: Ensure at least 10GB of free space on the root volume.
- [ ] **Virtual Environment Re-initialization**:
    ```bash
    rm -rf /root/.kalman_venv
    python3 -m venv /root/.kalman_venv
    ```

## 2. Dependency Upgrade Path
- [ ] **Core Dependencies**: Execute:
    ```bash
    /root/.kalman_venv/bin/pip install --upgrade torch numpy opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp pydantic
    ```
- [ ] **Optional Dev Dependencies**: Execute:
    ```bash
    /root/.kalman_venv/bin/pip install pytest hypothesis black ruff mypy
    ```

## 3. Post-Upgrade Verification
- [ ] **Test Execution**: Run the RoPE alignment verification script:
    ```bash
    /root/.kalman_venv/bin/python3 /root/test_rope_alignment.py
    ```
- [ ] **Telemetry Health Check**: Verify `OpenTelemetry` integration with a sample exporter trace.

## 4. Maintenance
- [ ] **Cache Cleanup**: Run `pip cache purge` to maintain disk space parity after installation.
