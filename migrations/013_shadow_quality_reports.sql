CREATE TABLE shadow_quality_feedback (
 request_id VARCHAR(36) PRIMARY KEY,
 verdict TEXT NOT NULL CHECK (verdict IN ('matched','better','worse')),
 primary_score DOUBLE PRECISION CHECK (primary_score BETWEEN 0 AND 1),
 judge_cost_cents DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK(judge_cost_cents>=0),
 shadow_score DOUBLE PRECISION CHECK (shadow_score BETWEEN 0 AND 1),
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE shadow_report_settings (
 id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(id),
 frequency TEXT NOT NULL DEFAULT 'daily' CHECK(frequency IN ('never','hourly','daily','weekly','biweekly','monthly','quarterly','yearly')),
 next_run_at TIMESTAMPTZ NOT NULL DEFAULT date_trunc('day',now()) + interval '1 day',
 last_run_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO shadow_report_settings(id) VALUES(TRUE);
CREATE TABLE shadow_quality_reports (
 id BIGSERIAL PRIMARY KEY, from_at TIMESTAMPTZ NOT NULL, to_at TIMESTAMPTZ NOT NULL,
 report JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
