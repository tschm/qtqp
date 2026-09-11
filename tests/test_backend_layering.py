# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""direct.py must not import its backends, but must still alias them."""

import ast
import inspect

import pytest

from qtqp import direct
from qtqp import solvers_dense
from qtqp import solvers_gpu
from qtqp import solvers_sparse

_BACKEND_MODULES = {
    "solvers_dense": solvers_dense,
    "solvers_gpu": solvers_gpu,
    "solvers_sparse": solvers_sparse,
}


def test_direct_has_no_module_level_backend_import():
  """The import cycle is gone: nothing pulls a backend in at import time."""
  tree = ast.parse(inspect.getsource(direct))
  imported = set()
  for node in tree.body:
    if isinstance(node, ast.ImportFrom) and node.module:
      imported.add(node.module.lstrip("."))
    elif isinstance(node, ast.Import):
      imported.update(alias.name for alias in node.names)
  assert not imported & set(_BACKEND_MODULES)


@pytest.mark.parametrize("name,module_name", sorted(direct._BACKEND_ALIASES.items()))
def test_backend_alias_resolves_to_the_owning_module(name, module_name):
  """``direct.ScipySolver`` and friends still resolve, to the one true class."""
  assert getattr(direct, name) is getattr(_BACKEND_MODULES[module_name], name)


def test_resolved_alias_is_cached_in_the_module_globals():
  """A resolved alias is a plain module attribute, so __getattr__ runs once."""
  direct.__dict__.pop("ScipySolver", None)
  assert direct.ScipySolver is solvers_sparse.ScipySolver
  assert direct.__dict__["ScipySolver"] is solvers_sparse.ScipySolver


def test_unknown_attribute_still_raises():
  """The lazy hook must not turn a typo into something importable."""
  with pytest.raises(AttributeError):
    direct.NoSuchSolver


def test_dir_lists_the_aliases():
  """The aliases are discoverable even before they are resolved."""
  assert set(direct._BACKEND_ALIASES) <= set(dir(direct))


def test_alias_is_usable_as_a_constructor():
  """The reported break: ``qtqp.direct.ScipySolver()`` must still build one."""
  backend = direct.ScipySolver()
  assert isinstance(backend, direct.LinearSolver)
