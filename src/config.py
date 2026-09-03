"""Column names, filters and constants used across the pipeline."""

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# The only columns this project reads. The BTS file ships 109; loading all of
# them costs about four times the memory for no gain.
USECOLS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "CRSElapsedTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

# A scheduled turnaround longer than this is treated as the aircraft going out
# of service rather than turning, so the pair is dropped from the chain.
MAX_TURN_MINUTES = 600

# Inbound-delay buckets used for the absorption curve. Half-open (lo, hi].
DELAY_BUCKETS = [
    (-10_000.0, 0.0, "early / on time"),
    (0.0, 15.0, "0-15"),
    (15.0, 30.0, "15-30"),
    (30.0, 60.0, "30-60"),
    (60.0, 120.0, "60-120"),
    (120.0, 10_000.0, "120+"),
]

# Buckets of scheduled turnaround time, used to test whether schedule buffer is
# what actually absorbs an inbound delay.
BUFFER_BUCKETS = [
    (0.0, 40.0, "<40 min"),
    (40.0, 60.0, "40-60"),
    (60.0, 90.0, "60-90"),
    (90.0, 150.0, "90-150"),
    (150.0, 600.0, "150+"),
]
