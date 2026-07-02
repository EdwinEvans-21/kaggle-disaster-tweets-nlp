def build_input_text(df, use_keyword=False):
    text = df["text"].fillna("")

    if not use_keyword:
        return text

    keyword = df["keyword"].fillna("").str.replace("%20", " ", regex=False)
    return keyword + " " + text
