# flake8: noqa

_BODO_EXPLANATION = """
DO NOT evaluate any intermediate DataFrame results unless absolutely necessary to answering the question. Instead use the head of the DataFrame or save to an intermediate result:

Good examples:
>>> df[df["A"] > 2].head(5)
OR
>>> _df = df[df["A"] > 2]
"""

PREFIX = f"""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. {_BODO_EXPLANATION}
You should use the tools below to answer the question posed of you:"""

MULTI_DF_PREFIX = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc."""+ f"{_BODO_EXPLANATION}"
"You should use the tools below to answer the question posed of you:"

SUFFIX_NO_DF = """
Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}

Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_MULTI_DF = """
This is the result of `print(df.head())` for each dataframe:
{dfs_head}

Begin!
Question: {input}
{agent_scratchpad}"""

PREFIX_FUNCTIONS = f"""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. {_BODO_EXPLANATION}"""

MULTI_DF_PREFIX_FUNCTIONS = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc.""" + f" {_BODO_EXPLANATION}"

FUNCTIONS_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}"""

FUNCTIONS_WITH_MULTI_DF = """
This is the result of `print(df.head())` for each dataframe:
{dfs_head}"""