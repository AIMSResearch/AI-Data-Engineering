with DAG("session_features", schedule="@hourly"):
    build_sessions = PythonOperator(task_id="build")
    rebuild_labels = PythonOperator(task_id="labels")
    build_sessions >> rebuild_labels

