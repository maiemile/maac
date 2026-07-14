# A FastAPI server to expose pymoo benchmark problems
# Original code by @light-weaver, RE problems added by @maiemile

from typing import Any

import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel
from pymoo.problems import get_problem
import utils as util

class PymooParameters(BaseModel):
    name: str
    n_vars: int = 2
    n_objs: int = 2

app = FastAPI()
re_problems = util.get_re_problems()


def get_pymoo_problem(name: str, n_vars: int, n_objs: int):
    """
    Get a pymoo problem instance by name, number of variables, and number of objectives.
    """

    problem = get_problem(name, n_var=n_vars, n_obj=n_objs)
    return problem


@app.get("/evaluate")
def evaluate(d: dict[str, list[float]], p: PymooParameters) -> dict[str, Any]:
    """
    Evaluate a pymoo problem instance with given parameters and input values.
    """
    problem = get_pymoo_problem(p.name, p.n_vars, p.n_objs)

    xs_df = pl.DataFrame(d)

    output = problem.evaluate(xs_df.to_numpy())
    output_df = pl.DataFrame(output, schema=[f"f_{i+1}" for i in range(problem.n_obj)])

    return d | output_df.to_dict(as_series=False)


@app.get("/evaluate_re")
def evaluate_re(d: dict[str, list[float]], p: PymooParameters) -> dict[str, Any]:
    """
    Evaluate a RE problem instance with given parameters and input values.
    """
    problem = re_problems[p.name]()
    
    xs_df = pl.DataFrame(d)
    xs_numpy = xs_df.to_numpy()
    output = [problem.evaluate(xs_numpy[i]) for i in range(len(xs_numpy))]
    output_df = pl.DataFrame(output, schema=[f"f_{i+1}" for i in range(problem.n_objectives)])
    return d | output_df.to_dict(as_series=False)


@app.get("/test")
def test():
    return 1


def start_server():
    import uvicorn

    uvicorn.run(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
