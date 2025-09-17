from typing import Any, Optional

import bodo.pandas as bd
import pandas as pd
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

from langchain_bodo.agent_toolkits import create_bodo_dataframes_agent


class FakeLLM(LLM):

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
           return prompt

    def _llm_type(self) -> str:
        return "fake"


def test_create_bodo_dataframes_agent():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]})
    bdf = bd.from_pandas(df2)

    llm = FakeLLM()

    with pytest.raises(ValueError):
        create_bodo_dataframes_agent(llm, df1, allow_dangerous_code=False)

    with pytest.raises(ValueError):
        create_bodo_dataframes_agent(llm, [1,2,3], allow_dangerous_code=True)

    create_bodo_dataframes_agent(llm, df1, allow_dangerous_code=True)
    create_bodo_dataframes_agent(llm, [bdf, df1], allow_dangerous_code=True)


