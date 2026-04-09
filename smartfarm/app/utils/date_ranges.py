from datetime import date, timedelta


def add_months(src_date: date, months: int) -> date:
    """
    외부 라이브러리 없이 월 더하기
    """
    month = src_date.month - 1 + months
    year = src_date.year + month // 12
    month = month % 12 + 1

    # 각 월의 마지막 일 처리
    day = min(src_date.day, [
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ][month - 1])

    return date(year, month, day)


def split_date_ranges(start_date: date, end_date: date, months_per_chunk: int = 6):
    """
    start_date ~ end_date를 6개월 단위 구간으로 분리
    """
    ranges = []
    current_start = start_date

    while current_start <= end_date:
        next_start = add_months(current_start, months_per_chunk)
        current_end = next_start - timedelta(days=1)

        if current_end > end_date:
            current_end = end_date

        ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return ranges