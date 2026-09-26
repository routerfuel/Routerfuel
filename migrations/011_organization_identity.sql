-- A stable tenant identity independent of API-key rotation. Existing keys keep
-- their historical key hash as the organization ID until explicitly grouped.
ALTER TABLE client_tiers ADD COLUMN organization_id VARCHAR(128);
UPDATE client_tiers SET organization_id = client_id WHERE organization_id IS NULL;

CREATE OR REPLACE FUNCTION default_client_organization_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.organization_id IS NULL THEN
        NEW.organization_id = NEW.client_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER client_tiers_default_organization
    BEFORE INSERT ON client_tiers
    FOR EACH ROW EXECUTE FUNCTION default_client_organization_id();

ALTER TABLE client_tiers ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE client_tiers ADD CONSTRAINT client_tiers_organization_id_nonempty
    CHECK (length(trim(organization_id)) > 0);
CREATE INDEX idx_client_tiers_organization ON client_tiers (organization_id);

COMMENT ON COLUMN client_tiers.organization_id IS
    'Stable tenant ID shared by multiple client key hashes. Set it explicitly when provisioning rotated or additional keys; never derive authorization scope from client_name or notes.';
