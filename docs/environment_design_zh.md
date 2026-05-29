# Future V2V Benchmark Environment 说明

## 路网与数据源

当前默认主环境为 `tlc_manhattan`：

- 订单到达、OD 热点、行程时长和价格强度来自 NYC TLC 黄出租车月度数据。
- 路网层级使用 Manhattan taxi zone，不使用 16 区合成网格作为主环境。
- travel time 优先使用同月黄出租 OD 在对应时段的中位行程时间。
- 缺失 OD 会回退到同 OD 全时段中位数、同 origin 中位数或全局中位数。
- 黄出租数据不被直接解释为 V2V 交易，只作为未来 V2V 需求与城市移动模式的真实校准源。

合成 16 区网格仍保留为 smoke/test fallback：当 `scale=smoke` 且 TLC 缓存不存在时，可以继续运行最小工程验证。

## 实体属性

订单只保留必要属性：

- `arrival_tick`：订单到达时刻。
- `origin_zone`：服务起点区域。
- `destination_zone`：服务完成后的目的区域。
- `demand_kwh`：需求电量。
- `max_wait_ticks`：最大等待窗口。
- `willingness_to_pay_per_kwh`：单位电量愿付价。
- `cancel_sensitivity`：等待过久后的取消敏感度。

车辆只保留必要属性：

- `join_tick` / `leave_tick`：入池和离池时刻。
- `current_zone` / `destination_zone`：当前位置和自身目的区域。
- `battery_capacity_kwh` / `current_soc_kwh` / `reserve_kwh`：电池容量、当前电量和保留电量。
- `reservation_price_per_kwh`：车主供电报价。
- `time_cost_per_min`：车主时间成本。
- `owner_accept_sensitivity`：接受概率敏感度。
- `fleet_flag`：是否为平台或半平台车辆。

## TLC 到 V2V 的改造口径

- `arrival_tick` 来自出租车 pickup 时间在 episode 窗口内的位置。
- `origin_zone` 来自 `PULocationID`，表示需求发生区域。
- `destination_zone` 来自 `DOLocationID`，在 V2V 中解释为服务完成后车辆可能靠近的活动区域。
- `demand_kwh` 由 trip distance、duration 和业务扰动生成，控制在 V2V 合理电量区间。
- `willingness_to_pay_per_kwh` 由 total amount 与电量需求校准，并加上下限约束。
- `max_wait_ticks` 不直接使用出租车等待语义，而是按时段压力生成。
- 车辆供给不直接使用同 tick 出租车作为供电车辆，而是用历史 dropoff/idle 空间分布校准入池区域，再按 V2V 业务生成电量、报价、在线时间和保留电量。

## 动态机制

- 订单按双峰需求到达，热点区域随时间旋转。
- 车辆按概率入池，fleet 车辆在线时间更长、报价更低。
- 路况随 tick 变化，早晚高峰会提高接驾时间。
- 等待越久，订单取消概率越高。
- 超过最大等待窗口的订单过期。
- 匹配成功后，车辆在接驾和服务期间忙碌，服务完成后可再次参与匹配。

## 约束匹配

候选边必须满足：

- 接驾时间不超过 `pickup_cap_minutes`。
- 车辆可供电量不低于订单需求。
- 车辆剩余在线时间覆盖接驾和服务承诺。
- 期望利润为正。

匹配目标为最大化候选边期望利润。第一版使用 Hungarian assignment，并允许订单不匹配。

为保证 Manhattan main 规模可训练，候选图默认按订单侧保留最近的若干可供电车辆，参数为 `environment.max_candidate_vehicles_per_order`。所有 baseline 和 RL 策略共享同一个候选图，因此比较口径一致。

## 主指标

主指标为 `future_v2v_score`，以平台利润为核心，同时惩罚：

- 服务率低于目标。
- 急单服务率低于目标。
- 过期率过高。
- 取消率过高。
- 平均接驾时间过长。
- 平均承诺时间过长。

这个指标用于表达未来 V2V 平台更真实的多目标经营约束，而不是只追求单边利润或单纯服务数量。
