{{ config(materialized='incremental', unique_key='session_id') }}
select session_id, max(event_time) as session_end
from {{ ref('raw_sessions') }}
{% if is_incremental() %}
where event_time >= (
  select coalesce(max(session_end), '1900-01-01')
  from {{ this }}
)
{% endif %}
group by 1

