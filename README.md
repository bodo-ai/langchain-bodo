# LangChain-Bodo

This package contains the LangChain integration with [Bodo DataFrames](https://github.com/bodo-ai/Bodo),
an open source, high performance DataFrame library that functions as a drop-in replacement for Pandas.

With just a single-line-of-code change, Bodo DataFrames automatically accelerates and scales Pandas code;

simply replace:
```py
import pandas as pd
```
with:
``` py
import bodo.pandas as pd
```

Under the hood, Bodo DataFrames uses lazy evaluation to optimize sequences of Pandas operations,
streams data through operators to enable processing larger-than-memory datasets, and
leverages MPI-based high-performance computing technology for efficient parallel execution that can
easily scale from laptop to large cluster.

## Installation

```bash
pip install -U langchain-bodo
```

No additional credentials/configurations are required.

## Agent Toolkits

> [!NOTE]
> Bodo DataFrames agents call the `Python` agent under the hood, which executes LLM generated Python code.
> Use with caution.

Bodo DataFrames agents are similar to [Pandas agents](https://python.langchain.com/docs/integrations/tools/pandas/)
except agent-generated code operates on Bodo DataFrames to answer questions on larger datasets.
Because Bodo DataFrames is mostly compatible with Pandas,
it is an ideal target for LLM code generation that's easy to verify, efficient, and scalable beyond the typical limitations of Pandas.

This example uses the Titanic dataset which can be found at https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv.

``` py
import bodo.pandas as pd
from langchain_bodo import create_bodo_dataframes_agent
from langchain_openai import OpenAI

df = pd.read_csv("titanic.csv")

agent = create_bodo_dataframes_agent(OpenAI(temperature=0), df, verbose=True)

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

You can also pass one or more Pandas DataFrames to agents:

``` py
from langchain_bodo import create_bodo_dataframes_agent
from langchain_openai import OpenAI
import pandas

df = pandas.read_csv("titanic.csv")
df2 = df.copy()

agent = create_bodo_dataframes_agent(OpenAI(temperature=0), [df, df2], verbose=True)
```

For more details refer to [Bodo DataFrames API documentation](https://docs.bodo.ai/latest/api_docs/dataframe_lib/).
