select l.order_id,
       l.label,
       f.feature_value
from labels l
left join features f
  on l.customer_id = f.customer_id
 and f.feature_ts <= l.label_ts
qualify row_number() over (
  partition by l.order_id
  order by f.feature_ts desc
) = 1

