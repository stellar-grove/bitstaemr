select
stat.game_date,
stat.game_pk,
stat.at_bat_number,
stat.pitch_type,
stat.pitch_name,
stat.release_speed,
stat.release_spin_rate,
stat.zone,
stat.type,
stat.hit_location,
stat.balls,
stat.strikes,
stat.outs_when_up
from
mlb.statcast stat

order by
stat.game_date desc,
stat.game_pk desc,
stat.at_bat_number,
stat.pitch_number