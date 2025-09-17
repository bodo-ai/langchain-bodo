# Langchain-Bodo

This package contains the Langchain integration with [Bodo DataFrames](https://github.com/bodo-ai/Bodo),
an open source, high performance DataFrame library that functions as a drop-in replacement for Pandas.

## Installation

```bash
pip install -U langchain-bodo
```

No additional credentials/configurations are required.

## Agent Toolkits

> [!NOTE]
> Bodo DataFrames agent calls the `Python` agent under the hood, which executes LLM generated Python code.
> Use with caution.

Bodo DataFrames agent is similar to the [Pandas DataFrame agents](https://python.langchain.com/docs/integrations/tools/pandas/)
except it converts Pandas DataFrames to Bodo DataFrames, which is ideal for compute intensive operations.

This example uses the titanic dataset which can be found [here]("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv").

``` py
import pandas as pd
from langchain_bodo import create_bodo_dataframes_agent
from langchain_openai import OpenAI

df = pd.read_csv("titanic.csv")

agent = create_bodo_dataframes_agent(OpenAI(temperature=0), df, verbose=True, allow_dangerous_code=True)

agent.invoke("What was the average age of the male passengers?")
```

Sample Output:
```
> Entering new AgentExecutor chain...
Thought: I need to filter the dataframe to only include male passengers and then calculate the average age.
Action: python_repl_ast
Action Input: df[df['Sex'] == 'male']['Age'].mean()30.72664459161148I now know the final answer
Final Answer: The average age of the male passengers is 30.73 years old.

> Finished chain.
```

You can also pass Bodo DataFrames directly:

``` py
import bodo.pandas as pd
from langchain-bodo import create_bodo_dataframes_agent
from langchain_openai import OpenAI

df = pd.read_csv("titanic.csv")

agent = create_bodo_dataframes_agent(OpenAI(temperature=0), df, verbose=True, allow_dangerous_code=True)
```

For more details refer to [Bodo DataFrames API documentation](https://docs.bodo.ai/latest/api_docs/dataframe_lib/).
