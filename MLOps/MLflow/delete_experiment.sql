DELETE FROM experiment_tags WHERE experiment_id=ANY(
    SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
);

DELETE FROM latest_metrics WHERE run_uuid=ANY(
    SELECT run_uuid FROM runs WHERE experiment_id=ANY(
        SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
    )
);

DELETE FROM metrics WHERE run_uuid=ANY(
    SELECT run_uuid FROM runs WHERE experiment_id=ANY(
        SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
    )
);

DELETE FROM tags WHERE run_uuid=ANY(
    SELECT run_uuid FROM runs WHERE experiment_id=ANY(
        SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
    )
);

DELETE FROM runs WHERE experiment_id=ANY(
    SELECT experiment_id FROM experiments where lifecycle_stage='deleted'
);

DELETE FROM experiments where lifecycle_stage='deleted';
