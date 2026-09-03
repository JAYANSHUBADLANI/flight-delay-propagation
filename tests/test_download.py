from src.download import month_url


def test_month_carries_no_leading_zero():
    # BTS serves 2024_1, not 2024_01. Zero-padding it 404s, and it is the
    # easiest way to break the download loop for months 1 through 9.
    assert month_url(2024, 1).endswith("_2024_1.zip")
    assert "2024_01" not in month_url(2024, 1)


def test_double_digit_months_are_unchanged():
    assert month_url(2024, 12).endswith("_2024_12.zip")


def test_url_points_at_the_bts_prezip_endpoint():
    url = month_url(2019, 7)
    assert url.startswith("https://transtats.bts.gov/PREZIP/")
    assert "On_Time_Reporting_Carrier_On_Time_Performance" in url
