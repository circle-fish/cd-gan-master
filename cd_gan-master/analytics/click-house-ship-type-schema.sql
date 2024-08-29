DROP TABLE IF EXISTS ship.type;
CREATE TABLE ship.type (
    data_source_reference String,
    primary_vessel_type String,
    specific_vessel_type String,
    ui_vessel_type String,
    data_source_raw_vessel_type String     
) ENGINE = Log;
