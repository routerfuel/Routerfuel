-- Snapshot the tenant on each request log. Do not replace client_id: that is
-- still the individual key hash used for revocation and key-level diagnostics.
ALTER TABLE request_logs ADD COLUMN organization_id VARCHAR(128);

-- Backfill through a correlated lookup so deleted/env-only keys retain their
-- historical key hash as a distinct tenant rather than becoming unscoped.
UPDATE request_logs AS logs
SET organization_id = COALESCE(
    (SELECT keys.organization_id FROM client_tiers AS keys WHERE keys.client_id = logs.client_id),
    logs.client_id
)
WHERE logs.client_id IS NOT NULL;

CREATE OR REPLACE FUNCTION set_request_log_organization_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.organization_id IS NULL AND NEW.client_id IS NOT NULL THEN
        SELECT organization_id INTO NEW.organization_id
        FROM client_tiers WHERE client_id = NEW.client_id;
        NEW.organization_id = COALESCE(NEW.organization_id, NEW.client_id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER request_logs_organization
    BEFORE INSERT ON request_logs
    FOR EACH ROW EXECUTE FUNCTION set_request_log_organization_id();

CREATE INDEX idx_rl_organization_created
    ON request_logs (organization_id, created_at DESC);

COMMENT ON COLUMN request_logs.organization_id IS
    'Tenant ID snapshotted at log insertion, stable across key rotation. NULL only for anonymous legacy requests. Do not infer authorization from client_name or request-supplied headers.';
