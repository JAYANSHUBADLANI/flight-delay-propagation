# Data

Nothing in this directory is committed. One month of BTS On-Time Performance is
about 235 MB uncompressed, and the whole of 2024 is roughly 2.8 GB.

## Getting it

```bash
make data                                   # Jan + Feb 2024, what this analysis used
make data-full                              # all twelve months of 2024
python3 scripts/download_data.py 2024 1     # just January 2024
python3 scripts/download_data.py 2019 7     # a 2019 month, for out-of-time checks
```

A month already present is skipped rather than re-fetched, so an interrupted run
is cheap to resume.

The files come from the US Bureau of Transportation Statistics as plain zips
over HTTPS, with no account, key or scraping involved:

```
https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip
```

The month carries no leading zero: `2024_1`, not `2024_01`. Data runs from 1987
to roughly two months behind the present.

To browse rather than script it:

- field selection and month picker:
  <https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr>
- table index:
  <https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD&QO_anzr=Nv&QO_fu146_anzr=b0-gvzr>

The dataset's formal name, for citation, is **Reporting Carrier On-Time
Performance (1987-present)**, published by the Bureau of Transportation
Statistics, US Department of Transportation.

## Sizes, so nothing surprises you

A month is 25-34 MB zipped and about 230 MB as CSV. The downloader pulls the
whole zip into memory before extracting, so **nothing appears in this directory
until a month finishes**, so an empty `data/raw/` part way through is not a stall.
The two months this analysis used are January 2024 (547,271 rows) and February
2024 (519,221).

## Why 2024

2020 and 2021 are unusable for this question: the pandemic collapsed schedules
and load factors, so turnaround buffers and propagation behave nothing like a
normal year. 2024 is the most recent full year of ordinary operations. 2019 is
kept as an out-of-time check rather than as training data.

## What each row is

One flight leg. The columns this project reads are listed in `src/config.py`;
the file ships 109 in total. The ones that matter most here are `Tail_Number`
(which makes aircraft rotations reconstructable at all), the scheduled and
actual departure and arrival times, and `LateAircraftDelay`, which is the
airline's own attribution of delay to a late inbound aircraft.
