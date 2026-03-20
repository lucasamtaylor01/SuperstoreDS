import pandas as pd

def cluster_profile(df):
    """
    Média das variáveis numéricas por cluster
    """
    return df.groupby("CLUSTER").mean(numeric_only=True)


def cluster_size(df):
    """
    Proporção de cada cluster
    """
    return df["CLUSTER"].value_counts(normalize=True)


def top_category(df, col):
    """
    Categoria mais frequente por cluster
    """
    return (
        df.groupby("CLUSTER")[col]
        .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
    )


def top2_categories(df, col):
    """
    Top 2 categorias por cluster
    """
    def top2(x):
        vc = x.value_counts()
        return pd.Series({
            "TOP1": vc.index[0] if len(vc) > 0 else None,
            "TOP2": vc.index[1] if len(vc) > 1 else None
        })

    return df.groupby("CLUSTER")[col].apply(top2).unstack()


def cluster_summary(df):
    """
    Resumo completo:
    - médias numéricas
    - tamanho do cluster
    """
    summary_num = df.groupby("CLUSTER").mean(numeric_only=True)
    summary_size = df["CLUSTER"].value_counts().rename("SIZE")

    return summary_num.join(summary_size)


def cluster_summary_full(df, categorical_cols):
    """
    Resumo completo incluindo:
    - numéricas
    - tamanho
    - categorias principais
    """
    summary = cluster_summary(df)

    for col in categorical_cols:
        top_cat = top_category(df, col)
        summary[col + "_TOP"] = top_cat

    return summary