# An Agent Swarm with a Mission

This is a fun project using the [OpenAI/swarm](https://github.com/openai/swarm) library with a local [Ollama](https://ollama.com/) model to accomplish an arbitrary mission.

It uses tools, can read web-pages, has a short-term and a long-term memory, can read and write files, evaluate code, etc.

It attempts to solve complex tasks as a team in a chaotic, unorganized manner.


## Installation

This code was developed with Python 3.12.7.

It is recommended to use a virtual environment such as venv or conda.

Install all required packages with `pip install -r requirements.txt`

We incorporated the swarm library directly into this project (under MIT license) because the source code is so slim and we needed a few small changes.



## Ollama

Download the Ollama model, e.g. `ollama run qwen2.5-coder:14b`

For this application, we need a larger context window of at least 48k.

Since we use the openai library for API access and it does not allow setting the context window, there is only this way to enlarge the context window:

### Change Ollama Model Context Size

```console
ollama show qwen2.5-coder:14b --modelfile > qwen2.5-coder-14.modelfile
```

Edit the model-file.
After the FROM line add
`PARAMETER num_ctx 49152`

Then import this model back to Ollama
```console
ollama create qwen2.5-coder-14b-48k --file qwen2.5-coder-14.modelfile
```

Check the import with `ollama list`
Check the memory usage with `ollama ps`

This setup (14b model with 48k context) requires less than 23 GB RAM and runs well on a rtx 4090.


## Configuration

Enter the **PROJECT_MISSION** and other settings in ***config.py***

You can edit the file ***.env*** to setup the LLM-API endpoint.
Or you can configure it to use Open-AI models with API-Key instead of local models.


## Run the system

Run `python run_swarm.py` to run the swarm.

Or simply start it from VS-Code, as we do.

The swarm creates a textfile (*.md) in the ***./outputs*** directory.

We open it with ***Sublime Text*** to follow the output with syntax highlighting.


### Start a new run from scratch

You can stop and continue a run, all settings are stored under ***.settings***

To start from scratch, simply enter a new MISSION or delete the ***.settings*** directory.


## Security remarks

Using AI agents with tools may cause this code to perform unwanted actions.

However, web access is limited to HTTP-Get, and the tool for evaluating Python code is disabled for the agents. We think this is safe.

If you want to provide additional capabilities to your swarm or create additional tools, you should always keep security in mind, e.g. run it in a sandbox.

This code is for research or educational purposes and is not intended for productive use. Use at your own risk.
