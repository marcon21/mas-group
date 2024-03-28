# TVA
> Team Members: ... and,Gabriel Graells.

## Usage

TVA program is contained in a Docker. First build the Docker.
```bash
docker build -t tva .
```
Run the TVA analysis by running the Docker Image. You need to mount the folder `input/`.
```bash
docker run -v $(pwd)/input:/input tva
```
Once the execution ends you should find a HTML file called `tva_report.html` with the analysis.

### Modify Voting Input
In the `input/` folder there is a `JSON` file named `voting_result.json` with a sample voting input.
```json
{
    "voting":
        [   # v0,  v1,  v2,  v3,  v4
            ["C", "B", "C", "B", "B"], # p0
            ["A", "D", "D", "D", "A"], # p1
            ["D", "C", "A", "C", "D"], # p2
            ["B", "A", "B", "A", "C"]  # p3
        ]
}
```
* **Columns** represent voters, $v_i$.
* **Row** is preferences, $p_j$.

In the example above the first voter $v_0$ has the following voting preference $\{C, A, D, B\}$ ordered from most to least prefered alternative.

**You can modify the voting matrix and re-run the Docker to test.**

---
# Developer Usage

## Python
The project uses the latest Python version (I think) **Python 3.10.12**

## requirements.txt
To ensure you have all the required Python libraries install all libraries in the `requirements.txt` file. (if you use `pip`):
```bash
pip install -r requirements.txt
```

## pre-commit
> *Pre-commit is a tool used in software development to automatically execute a set of checks or tasks on code before it is committed to version control, helping to maintain code quality and consistency.*

Check [pre-commit](https://pre-commit.com/) docs to install it.