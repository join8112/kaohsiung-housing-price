import math

import pandas as pd

from khp.clean import (chinese_to_int, extract_road, parse_floor, parse_roc_date,
                       parse_total_floors, parse_trade_counts, short_building_type)


def test_parse_roc_date():
    assert parse_roc_date("1130921") == pd.Timestamp("2024-09-21")
    assert parse_roc_date("0990105") == pd.Timestamp("2010-01-05")
    assert parse_roc_date("990105") == pd.Timestamp("2010-01-05")
    assert parse_roc_date("1050800") == pd.Timestamp("2016-08-01")  # day 00
    assert pd.isna(parse_roc_date(""))
    assert pd.isna(parse_roc_date("11313"))


def test_chinese_to_int():
    assert chinese_to_int("五") == 5
    assert chinese_to_int("十") == 10
    assert chinese_to_int("十二") == 12
    assert chinese_to_int("二十") == 20
    assert chinese_to_int("三十八") == 38
    assert chinese_to_int("一百零一") == 101
    assert chinese_to_int("12") == 12


def test_parse_floor():
    assert parse_floor("五層") == (5.0, 1, False)
    assert parse_floor("十二層，十三層") == (12.0, 2, False)
    assert parse_floor("地下一層，一層") == (1.0, 2, False)
    f, n, whole = parse_floor("全")
    assert math.isnan(f) and n == 0 and whole
    assert math.isnan(parse_floor("見其他登記事項")[0])


def test_parse_total_floors():
    assert parse_total_floors("十五層") == 15.0
    assert math.isnan(parse_total_floors(""))


def test_parse_trade_counts():
    assert parse_trade_counts("土地2建物1車位1") == (2.0, 1.0, 1.0)


def test_extract_road():
    assert extract_road("高雄市三民區建工路６０１巷１～３０號") == "建工路"
    assert extract_road("高雄市前鎮區中山二路１號") == "中山二路"
    assert extract_road("高雄市鳳山區光遠路一段２號") == "光遠路一段"
    assert extract_road("高雄市苓雅區四維三路") == "四維三路"
    assert extract_road("高雄市仁武區仁武里仁雄路１號") == "仁雄路"
    assert extract_road("高雄市鳳山區文英里１２鄰光遠路２號") == "光遠路"
    assert extract_road("高雄市大寮區大里路１號") == "大里路"
    assert extract_road("高雄市楠梓區德民段１２３地號") is None


def test_short_building_type():
    assert short_building_type("住宅大樓(11層含以上有電梯)") == "大樓"
    assert short_building_type("套房(1房1廳1衛)") == "套房"
    assert short_building_type("店面(店鋪)") is None
