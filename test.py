
import pathlib, importlib.util
from ament_index_python.packages import get_package_share_directory

pkg = "five_dof_cobot_roll_j4_moveit_config"
share = pathlib.Path(get_package_share_directory(pkg))
demo = share / "launch" / "demo.launch.py"

spec = importlib.util.spec_from_file_location("demo_launch", demo)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ld = mod.generate_launch_description()

def walk(entities, depth=0):
    for e in entities:
        # launch_ros.actions.Node
        if e.__class__.__name__ == "Node":
            pkgname = getattr(e, "_Node__package", None)
            execname = getattr(e, "_Node__executable", None)
            nodename = getattr(e, "_Node__node_name", None)
            params = getattr(e, "_Node__parameters", [])

            found = []
            for p in params:
                if isinstance(p, dict):
                    for k, v in p.items():
                        if isinstance(v, tuple):
                            found.append((k, v))
                # parameters list 안에 tuple 자체가 들어간 경우도 잡기
                if isinstance(p, tuple):
                    found.append(("__param_item__", p))

            if found:
                print(f"\n### Node pkg={pkgname} exec={execname} name={nodename}")
                for k, v in found:
                    print(f"  >>> TUPLE FOUND: {k} = {v}")

        sub = getattr(e, "entities", None)
        if sub:
            walk(sub, depth + 1)

walk(ld.entities)
