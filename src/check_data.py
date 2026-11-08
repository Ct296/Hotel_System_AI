import re
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvi import ViTokenizer
from transformers import AutoTokenizer

from config import DATASET_FILE, METRICS_DIR, MODEL_NAME
from clean_data import clean_text, load_abbreviations


def suggest_max_len(p99_len, max_len_observed):
    """Gợi ý MAX_LEN chuẩn dựa trên phân vị 99% của PhoBERT token (Tối đa của PhoBERT là 256)."""
    target = max(p99_len, min(max_len_observed, 256))
    for candidate in [64, 96, 128, 160, 192, 224, 256]:
        if candidate >= target:
            return candidate
    return 256


def run_data_audit():
    if not DATASET_FILE.exists():
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {DATASET_FILE}")
        return

    print("⏳ Đang đọc dữ liệu gốc và khởi tạo PhoBERT Tokenizer để đo đạc...")
    with open(DATASET_FILE, "r", encoding="utf-8-sig") as f:
        content = f.read()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    abbrev_dict = load_abbreviations()

    # Regex bóc tách từng block #ID
    entry_pattern = re.compile(r"#(\d+)\s*(.*?)\s*(?=(?:#\d+)|\Z)", re.DOTALL)
    label_pattern = re.compile(r"\{([^,]+),\s*(positive|negative|neutral)\}", re.IGNORECASE)
    html_error_pattern = re.compile(r"&[a-zA-Z]+;?")

    entries = []
    empty_text_ids = []        # Có nhãn nhưng trống comment
    empty_label_ids = []       # Có comment nhưng trống nhãn
    html_error_ids = []        # Bị dính mã HTML (&atilde...)
    conflict_aspect_ids = []   # Cùng 1 câu mà 1 khía cạnh bị gán 2 lần
    text_seen = defaultdict(list)

    aspect_counter = Counter()
    aspect_polarity_counter = defaultdict(lambda: Counter())
    polarity_total_counter = Counter()
    overall_sentiment_counter = Counter()

    raw_word_lengths = []
    pyvi_word_lengths = []
    phobert_token_lengths = []
    aspects_per_comment = []

    for match in entry_pattern.finditer(content):
        entry_id = match.group(1)
        block = match.group(2).strip()

        raw_labels = label_pattern.findall(block)
        raw_text = label_pattern.sub("", block).strip()
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        # 1. Kiểm tra lỗi trống văn bản hoặc trống nhãn
        if not raw_text and raw_labels:
            empty_text_ids.append(entry_id)
        if raw_text and not raw_labels:
            empty_label_ids.append(entry_id)
        if not raw_text and not raw_labels:
            empty_text_ids.append(entry_id)
            empty_label_ids.append(entry_id)

        # 2. Kiểm tra lỗi HTML entity trong văn bản gốc
        if html_error_pattern.search(raw_text):
            html_error_ids.append((entry_id, raw_text[:80]))

        # 3. Kiểm tra trùng lặp nội dung bình luận
        if raw_text:
            text_seen[raw_text.lower()].append(entry_id)

        # 4. Kiểm tra mâu thuẫn/lặp khía cạnh trong cùng 1 câu
        seen_aspects_in_entry = {}
        sentiments_in_entry = []
        for asp, pol in raw_labels:
            asp = asp.strip().upper()
            pol = pol.strip().lower()
            if asp in seen_aspects_in_entry:
                conflict_aspect_ids.append((entry_id, asp, seen_aspects_in_entry[asp], pol))
            seen_aspects_in_entry[asp] = pol

            aspect_counter[asp] += 1
            aspect_polarity_counter[asp][pol] += 1
            polarity_total_counter[pol] += 1
            sentiments_in_entry.append(pol)

        # Tính sắc thái tổng thể của câu
        pos_c = sentiments_in_entry.count("positive")
        neg_c = sentiments_in_entry.count("negative")
        overall = "positive" if pos_c > neg_c else ("negative" if neg_c > pos_c else "neutral")
        overall_sentiment_counter[overall] += 1

        # 5. Đo độ dài câu (chỉ tính trên câu có chữ)
        if raw_text:
            cleaned = clean_text(raw_text, abbrev_dict)
            segmented = ViTokenizer.tokenize(cleaned)
            # Đếm số lượng token thực sự khi đưa vào PhoBERT (kể cả <s> và </s>)
            token_ids = tokenizer.encode(segmented, truncation=False)

            raw_len = len(raw_text.split())
            pyvi_len = len(segmented.split())
            bpe_len = len(token_ids)

            raw_word_lengths.append((raw_len, entry_id, raw_text))
            pyvi_word_lengths.append((pyvi_len, entry_id))
            phobert_token_lengths.append((bpe_len, entry_id, raw_text))
            aspects_per_comment.append((len(raw_labels), entry_id, raw_text))

        entries.append(entry_id)

    # Xử lý thống kê số học
    raw_lens = np.array([x[0] for x in raw_word_lengths])
    pyvi_lens = np.array([x[0] for x in pyvi_word_lengths])
    bpe_lens = np.array([x[0] for x in phobert_token_lengths])
    asp_counts = np.array([x[0] for x in aspects_per_comment])

    longest_raw = max(raw_word_lengths, key=lambda x: x[0])
    shortest_raw = min(raw_word_lengths, key=lambda x: x[0])
    longest_bpe = max(phobert_token_lengths, key=lambda x: x[0])
    most_aspects_entry = max(aspects_per_comment, key=lambda x: x[0])

    duplicate_groups = {txt: ids for txt, ids in text_seen.items() if len(ids) > 1}

    p90_bpe = int(np.percentile(bpe_lens, 90))
    p95_bpe = int(np.percentile(bpe_lens, 95))
    p99_bpe = int(np.percentile(bpe_lens, 99))
    p995_bpe = int(np.percentile(bpe_lens, 99.5))
    recommended_max_len = suggest_max_len(p995_bpe, longest_bpe[0])

    # Đếm tỷ lệ bao phủ ở các mốc MAX_LEN phổ biến
    coverage_stats = {}
    for m_len in [64, 128, 160, 192, 256]:
        cov = np.mean(bpe_lens <= m_len) * 100
        truncated_cnt = int(np.sum(bpe_lens > m_len))
        coverage_stats[m_len] = (cov, truncated_cnt)

    # Khía cạnh hiếm (dưới 15 mẫu trên toàn bộ 3000 câu)
    rare_pairs = []
    total_valid_pairs = 0
    for asp in sorted(aspect_polarity_counter.keys()):
        for pol in ["positive", "negative", "neutral"]:
            cnt = aspect_polarity_counter[asp][pol]
            if cnt > 0:
                total_valid_pairs += 1
                if cnt < 15:
                    rare_pairs.append((f"{asp}__{pol}", cnt))

    # ==================== XUẤT BÁO CÁO ====================
    lines = [
        "=" * 75,
        "📊 BÁO CÁO KIỂM ĐỊNH & THỐNG KÊ DỮ LIỆU (EXPLORATORY DATA ANALYSIS)",
        "=" * 75,
        f"1. KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU (DATA INTEGRITY):",
        f"   - Tổng số mẫu (comment) quét được       : {len(entries)} mẫu",
        f"   - Số mẫu CÓ NHÃN nhưng TRỐNG COMMENT    : {len(empty_text_ids)} mẫu "
        f"{f'(ID: {empty_text_ids})' if empty_text_ids else '(Không có - Rất tốt ✅)'}",
        f"   - Số mẫu CÓ COMMENT nhưng TRỐNG NHÃN    : {len(empty_label_ids)} mẫu "
        f"{f'(ID: {empty_label_ids})' if empty_label_ids else '(Không có - Rất tốt ✅)'}",
        f"   - Số mẫu bị dính mã lỗi HTML (&atilde..): {len(html_error_ids)} mẫu "
        f"{f'(VD ID: {[x[0] for x in html_error_ids[:5]]})' if html_error_ids else '(Không có)'}",
        f"   - Số mẫu gán trùng/mâu thuẫn khía cạnh  : {len(conflict_aspect_ids)} trường hợp "
        f"{f'(VD: {conflict_aspect_ids[:3]})' if conflict_aspect_ids else '(Không có ✅)'}",
        f"   - Số cụm bình luận trùng lặp nội dung   : {len(duplicate_groups)} cụm",
        "",
        "-" * 75,
        "2. THỐNG KÊ ĐỘ DÀI BÌNH LUẬN (CƠ SỞ KHOA HỌC ĐỂ CHỌN MAX_LEN):",
        f"   a) Theo số âm tiết (Từ đơn tách theo khoảng trắng):",
        f"      + Ngắn nhất : {shortest_raw[0]} từ (Câu #{shortest_raw[1]}: \"{shortest_raw[2]}\")",
        f"      + Dài nhất  : {longest_raw[0]} từ (Câu #{longest_raw[1]}: \"{longest_raw[2][:70]}...\")",
        f"      + Trung bình: {raw_lens.mean():.1f} ± {raw_lens.std():.1f} từ | Trung vị (Median): {np.median(raw_lens):.0f} từ",
        "",
        f"   b) Theo từ ghép tiếng Việt (Sau khi qua PyVi ViTokenizer):",
        f"      + Ngắn nhất : {pyvi_lens.min()} từ ghép | Dài nhất: {pyvi_lens.max()} từ ghép | Trung bình: {pyvi_lens.mean():.1f} từ ghép",
        "",
        f"   c) Theo Subword Token thực tế của PhoBERT (Đầu vào trực tiếp của mạng):",
        f"      + Ngắn nhất : {bpe_lens.min()} tokens | Dài nhất: {longest_bpe[0]} tokens (Câu #{longest_bpe[1]})",
        f"      + Trung bình: {bpe_lens.mean():.1f} tokens | Trung vị: {np.median(bpe_lens):.0f} tokens",
        f"      + Phân vị 90% (90th percentile): {p90_bpe} tokens",
        f"      + Phân vị 95% (95th percentile): {p95_bpe} tokens",
        f"      + Phân vị 99% (99th percentile): {p99_bpe} tokens",
        f"      + Phân vị 99.5%                : {p995_bpe} tokens",
        "",
        "   d) Bảng độ phủ dữ liệu theo từng mốc MAX_LEN (Trả lời phản biện hội đồng):",
    ]

    for m_len, (cov, trunc_cnt) in coverage_stats.items():
        lines.append(
            f"      - MAX_LEN = {m_len:3d} -> Bao phủ toàn vẹn {cov:6.2f}% dữ liệu (Chỉ có {trunc_cnt:3d}/{len(entries)} câu bị cắt bớt)"
        )

    lines.extend([
        f"\n   => 💡 KẾT LUẬN ĐỀ XUẤT: Chọn MAX_LEN = {recommended_max_len} (Bao phủ >= 99.5% câu mà không lãng phí bộ nhớ đệm).",
        "",
        "-" * 75,
        "3. THỐNG KÊ MẬT ĐỘ KHÍA CẠNH TRÊN MỖI BÌNH LUẬN:",
        f"   - Tổng số lượt nhãn (aspect-polarity)   : {int(asp_counts.sum())} lượt nhãn",
        f"   - Trung bình số khía cạnh / 1 bình luận : {asp_counts.mean():.2f} khía cạnh/câu",
        f"   - Ít nhất  : {asp_counts.min()} khía cạnh | Nhiều nhất: {most_aspects_entry[0]} khía cạnh (Câu #{most_aspects_entry[1]})",
        f"   - Phân phối số lượng khía cạnh trong 1 câu:",
    ])

    asp_dist = Counter(asp_counts)
    for k in sorted(asp_dist.keys()):
        pct = asp_dist[k] / len(entries) * 100
        lines.append(f"      + Câu có {k:2d} khía cạnh: {asp_dist[k]:4d} câu ({pct:5.2f}%)")

    lines.extend([
        "",
        "-" * 75,
        "4. THỐNG KÊ PHÂN PHỐI CẢM XÚC & BẤT CÂN BẰNG DỮ LIỆU (CLASS IMBALANCE):",
        "   a) Phân phối Cảm xúc tổng thể (Overall Sentiment của cả câu):",
        f"      + Positive (Tích cực)  : {overall_sentiment_counter['positive']:4d} câu ({overall_sentiment_counter['positive']/len(entries)*100:5.2f}%)",
        f"      + Negative (Tiêu cực)  : {overall_sentiment_counter['negative']:4d} câu ({overall_sentiment_counter['negative']/len(entries)*100:5.2f}%)",
        f"      + Neutral  (Trung tính): {overall_sentiment_counter['neutral']:4d} câu ({overall_sentiment_counter['neutral']/len(entries)*100:5.2f}%)",
        "",
        "   b) Phân phối Cảm xúc tính trên tổng số lượt khía cạnh:",
        f"      + Lượt khen (Positive) : {polarity_total_counter['positive']:4d} lượt ({polarity_total_counter['positive']/asp_counts.sum()*100:5.2f}%)",
        f"      + Lượt chê  (Negative) : {polarity_total_counter['negative']:4d} lượt ({polarity_total_counter['negative']/asp_counts.sum()*100:5.2f}%)",
        f"      + Trung lập (Neutral)  : {polarity_total_counter['neutral']:4d} lượt ({polarity_total_counter['neutral']/asp_counts.sum()*100:5.2f}%)",
        "",
        f"   c) Thống kê độ hiếm của tổ hợp nhãn (Giải thích tại sao dùng Micro-F1 & Weighted-F1):",
        f"      + Tổng số khía cạnh gốc              : {len(aspect_counter)} khía cạnh",
        f"      + Tổng số tổ hợp (Aspect + Polarity) : {total_valid_pairs} tổ hợp có xuất hiện",
        f"      + Số tổ hợp SIÊU HIẾM (< 15 mẫu/3000 câu): {len(rare_pairs)}/{total_valid_pairs} tổ hợp ({len(rare_pairs)/total_valid_pairs*100:.1f}%)",
        "",
        "   d) Bảng chi tiết toàn bộ các Khía cạnh (Sắp xếp từ phổ biến nhất đến hiếm nhất):",
        f"      {'STT':<4} {'TÊN KHÍA CẠNH':<35} {'TỔNG':>6} | {'POS':>5} {'NEG':>5} {'NEU':>5}",
        "      " + "-" * 65,
    ])

    for idx, (asp, total_cnt) in enumerate(aspect_counter.most_common(), 1):
        p_cnt = aspect_polarity_counter[asp]["positive"]
        ng_cnt = aspect_polarity_counter[asp]["negative"]
        nu_cnt = aspect_polarity_counter[asp]["neutral"]
        lines.append(f"      {idx:<4} {asp:<35} {total_cnt:>6} | {p_cnt:>5} {ng_cnt:>5} {nu_cnt:>5}")

    lines.append("=" * 75)
    report_str = "\n".join(lines)
    print("\n" + report_str)

    # Lưu ra file báo cáo
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = METRICS_DIR / "data_audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n📄 Đã xuất báo cáo kiểm định dữ liệu ra file: {report_path}")

    # Vẽ biểu đồ phân phối độ dài Token và Số lượng khía cạnh để đưa vào Luận văn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(bpe_lens, bins=40, color="#2b5c8f", edgecolor="white", alpha=0.85)
    ax1.axvline(p95_bpe, color="orange", linestyle="--", linewidth=2, label=f"95% ({p95_bpe} tokens)")
    ax1.axvline(p99_bpe, color="red", linestyle="--", linewidth=2, label=f"99% ({p99_bpe} tokens)")
    ax1.set_title("Phân phối độ dài Token PhoBERT trên tập dữ liệu")
    ax1.set_xlabel("Số lượng Subword Tokens / Bình luận")
    ax1.set_ylabel("Số lượng bình luận")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend()

    keys = sorted(asp_dist.keys())
    vals = [asp_dist[k] for k in keys]
    ax2.bar([str(k) for k in keys], vals, color="#388e3c", edgecolor="white", alpha=0.85)
    ax2.set_title("Phân phối số lượng Khía cạnh trong 1 Bình luận")
    ax2.set_xlabel("Số khía cạnh / Bình luận")
    ax2.set_ylabel("Số lượng bình luận")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = METRICS_DIR / "data_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"📊 Đã lưu biểu đồ phân phối dữ liệu tại: {chart_path}")


if __name__ == "__main__":
    run_data_audit()