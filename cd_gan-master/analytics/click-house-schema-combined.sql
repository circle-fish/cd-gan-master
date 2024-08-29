CREATE DATABASE IF NOT EXISTS ship;

DROP TABLE IF EXISTS ship.tank_sp_f;
CREATE TABLE ship.tank_sp_f (
    time_utc DateTime,
    longitude Float64,
    latitude Float64,
    mmsi_hash String,
    spedd Float64,
    heading Float64,    
    flag_hash String,
    breadth_extream UInt32,
    draught_large UInt32,
    grt UInt32,
    dwt UInt32,    
    ship_type_code String,
    filename String,
    primary_vessel_type String,
    ui_vessel_type String
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(time_utc)
ORDER BY (ship_type_code, time_utc)


y5J7Tnd4
