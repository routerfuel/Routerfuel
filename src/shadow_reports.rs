use crate::admin::AdminState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;

pub fn interval(frequency: &str) -> Option<&'static str> {
    match frequency {
        "hourly" => Some("1 hour"),
        "daily" => Some("1 day"),
        "weekly" => Some("7 days"),
        "biweekly" => Some("14 days"),
        "monthly" => Some("1 month"),
        "quarterly" => Some("3 months"),
        "yearly" => Some("1 year"),
        "never" => Some("1 day"),
        _ => None,
    }
}
#[derive(Deserialize)]
pub struct Settings {
    pub frequency: String,
}
pub async fn settings(State(s): State<AdminState>) -> Response {
    match sqlx::query("SELECT frequency,next_run_at::text AS next_run_at FROM shadow_report_settings WHERE id=TRUE").fetch_one(s.pool.as_ref()).await {
        Ok(r)=>Json(json!({"frequency":r.get::<String,_>("frequency"),"next_run_at":r.get::<String,_>("next_run_at")})).into_response(),Err(_)=>StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}
pub async fn set_settings(State(s): State<AdminState>, Json(v): Json<Settings>) -> Response {
    let Some(period) = interval(&v.frequency) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"Unsupported frequency"})),
        )
            .into_response();
    };
    match sqlx::query("UPDATE shadow_report_settings SET frequency=$1,next_run_at=now()+$2::interval WHERE id=TRUE").bind(&v.frequency).bind(period).execute(s.pool.as_ref()).await {Ok(_)=>Json(json!({"frequency":v.frequency})).into_response(),Err(_)=>StatusCode::INTERNAL_SERVER_ERROR.into_response()}
}
#[derive(Deserialize)]
pub struct Feedback {
    request_id: String,
    verdict: String,
    primary_score: Option<f64>,
    shadow_score: Option<f64>,
}
pub async fn feedback(State(s): State<AdminState>, Json(v): Json<Feedback>) -> Response {
    if !matches!(v.verdict.as_str(), "matched" | "better" | "worse")
        || [v.primary_score, v.shadow_score]
            .into_iter()
            .flatten()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
    {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let result=sqlx::query("INSERT INTO shadow_quality_feedback(request_id,verdict,primary_score,shadow_score) SELECT $1,$2,$3,$4 WHERE EXISTS(SELECT 1 FROM shadow_comparisons WHERE request_id=$1 AND shadow_error IS NULL) ON CONFLICT(request_id) DO UPDATE SET verdict=excluded.verdict,primary_score=excluded.primary_score,shadow_score=excluded.shadow_score")
        .bind(&v.request_id).bind(&v.verdict).bind(v.primary_score).bind(v.shadow_score).execute(s.pool.as_ref()).await;
    match result {
        Ok(r) if r.rows_affected() > 0 => StatusCode::NO_CONTENT.into_response(),
        Ok(_) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
pub async fn reports(State(s): State<AdminState>) -> Response {
    match sqlx::query("SELECT id,from_at::text AS from_at,to_at::text AS to_at,report FROM shadow_quality_reports ORDER BY id DESC LIMIT 100").fetch_all(s.pool.as_ref()).await {
        Ok(rows)=>Json(rows.iter().map(|r|json!({"id":r.get::<i64,_>("id"),"from":r.get::<String,_>("from_at"),"to":r.get::<String,_>("to_at"),"report":r.get::<Value,_>("report")})).collect::<Vec<_>>()).into_response(),Err(_)=>StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}
pub async fn run_due(pool: &sqlx::PgPool) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    let row=sqlx::query("SELECT frequency,last_run_at,next_run_at FROM shadow_report_settings WHERE id=TRUE AND frequency<>'never' AND next_run_at<=now() FOR UPDATE SKIP LOCKED").fetch_optional(&mut *tx).await?;
    if let Some(r) = row {
        let from: chrono::DateTime<chrono::Utc> = r.get("last_run_at");
        let to = chrono::Utc::now();
        let rows=sqlx::query("SELECT c.primary_model,c.shadow_model,count(*) AS comparisons,count(*) FILTER(WHERE c.shadow_error IS NOT NULL) AS failures,count(*) FILTER(WHERE f.verdict='matched') AS matched,count(*) FILTER(WHERE f.verdict='better') AS better,count(*) FILTER(WHERE f.verdict='worse') AS worse,count(*) FILTER(WHERE f.request_id IS NULL AND c.shadow_error IS NULL) AS unevaluated,COALESCE(sum(c.primary_cost_cents-c.shadow_cost_cents) FILTER(WHERE f.verdict IN ('matched','better')),0)::float8 AS evaluated_savings_cents,COALESCE(sum(c.shadow_cost_cents),0)::float8 + COALESCE(sum(f.judge_cost_cents),0)::float8 AS experiment_spend_cents FROM shadow_comparisons c LEFT JOIN shadow_quality_feedback f USING(request_id) WHERE c.created_at >= $1 AND c.created_at < $2 GROUP BY c.primary_model,c.shadow_model")
            .bind(from).bind(to).fetch_all(&mut *tx).await?;
        let report=rows.iter().map(|r|{let mut v=json!({"primary_model":r.get::<String,_>("primary_model"),"shadow_model":r.get::<String,_>("shadow_model")});for key in ["comparisons","failures","matched","better","worse","unevaluated"]{v[key]=json!(r.get::<i64,_>(key));}for key in ["evaluated_savings_cents","experiment_spend_cents"]{v[key]=json!(r.get::<f64,_>(key));}v}).collect::<Vec<_>>();
        sqlx::query("INSERT INTO shadow_quality_reports(from_at,to_at,report) VALUES($1,$2,$3)")
            .bind(from)
            .bind(to)
            .bind(json!(report))
            .execute(&mut *tx)
            .await?;
        let frequency: String = r.get("frequency");
        sqlx::query("UPDATE shadow_report_settings SET last_run_at=$1,next_run_at=$1+$2::interval WHERE id=TRUE").bind(to).bind(interval(&frequency).unwrap()).execute(&mut *tx).await?;
    }
    tx.commit().await
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn report_frequencies() {
        assert_eq!(interval("biweekly"), Some("14 days"));
        assert_eq!(interval("monthly"), Some("1 month"));
        assert!(interval("never").is_some());
        assert!(interval("bad").is_none());
    }
}
#[cfg(test)]
mod database_tests {
    use super::*;
    #[tokio::test]
    #[ignore = "requires disposable REPORT_TEST_DATABASE_URL"]
    async fn scheduled_report_is_idempotent_and_counts_quality() {
        let pool = sqlx::PgPool::connect(&std::env::var("REPORT_TEST_DATABASE_URL").unwrap())
            .await
            .unwrap();
        sqlx::query("UPDATE shadow_report_settings SET frequency='daily',last_run_at=now()-interval '1 day',next_run_at=now()-interval '1 minute'").execute(&pool).await.unwrap();
        run_due(&pool).await.unwrap();
        let first: i64 = sqlx::query_scalar("SELECT count(*) FROM shadow_quality_reports")
            .fetch_one(&pool)
            .await
            .unwrap();
        run_due(&pool).await.unwrap();
        let second: i64 = sqlx::query_scalar("SELECT count(*) FROM shadow_quality_reports")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(first, second);
        let report: Value = sqlx::query_scalar(
            "SELECT report FROM shadow_quality_reports ORDER BY id DESC LIMIT 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(report[0]["matched"], 1);
        assert_eq!(report[0]["evaluated_savings_cents"], 1.0);
        assert_eq!(report[0]["experiment_spend_cents"], 1.1);
        sqlx::query("INSERT INTO request_logs(request_id,provider,model_name,latency_ms) SELECT 'measured-'||n,'anthropic','claude-haiku-5-5',1234 FROM generate_series(1,20) n ON CONFLICT(request_id) DO NOTHING").execute(&pool).await.unwrap();
        let engine = crate::route_engine::RouteEngine::new();
        engine.refresh_measurements(&pool).await.unwrap();
        assert_eq!(engine.find("claude-haiku-5-5").unwrap().latency_ms, 1234);
        sqlx::query("DELETE FROM request_logs WHERE request_id LIKE 'measured-%'")
            .execute(&pool)
            .await
            .unwrap();
        engine.refresh_measurements(&pool).await.unwrap();
        assert_eq!(engine.find("claude-haiku-5-5").unwrap().latency_ms, 90);
        sqlx::query("UPDATE shadow_report_settings SET frequency='never',next_run_at=now()-interval '1 day'").execute(&pool).await.unwrap();
        run_due(&pool).await.unwrap();
        let count: i64 = sqlx::query_scalar("SELECT count(*) FROM shadow_quality_reports")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(second, count);
    }
}
