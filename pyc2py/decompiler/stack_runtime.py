import ast
from abc import ABC

from pyc2py.decompiler.context import DecompilerContext


class StackRuntimeMixin(DecompilerContext, ABC):
    def push_empty_list_under_top(self) -> None:
        # PyPy starts a list comprehension with the iterable left on top
        top = self.pop_or_none()
        self.stack.append(ast.List(elts=[], ctx=ast.Load()))
        self.stack.append(top)

    def duplicate_top(self) -> None:
        if not self.stack:
            self.warnings.append("DUP_TOP on empty stack")
            self.stack.append(ast.Constant(value=None))
            return

        self.stack.append(self.stack[-1])

    def duplicate_top_two(self) -> None:
        if len(self.stack) < 2:
            self.warnings.append("DUP_TOP_TWO on shallow stack")
            self.stack.append(ast.Constant(value=None))
            self.stack.append(ast.Constant(value=None))
            return

        first, second = self.stack[-2], self.stack[-1]
        self.stack.append(first)
        self.stack.append(second)

    def duplicate_top_items(self, count: int) -> None:
        # DUP_TOPX before 3.2
        if count < 1 or len(self.stack) < count:
            self.warnings.append(f"DUP_TOPX {count} on shallow stack")

            for _ in range(max(count, 0)):
                self.stack.append(ast.Constant(value=None))

            return

        for _ in range(count):
            self.stack.copy_from_top(count - 1)

    def copy_stack_item(self, opcode_depth: int) -> None:
        if opcode_depth < 1:
            self.warnings.append(f"COPY with invalid depth: {opcode_depth}")
            self.stack.append(ast.Constant(value=None))
            return

        depth = opcode_depth - 1
        if len(self.stack) <= depth:
            self.warnings.append(f"COPY_{opcode_depth} on shallow stack")
            self.stack.append(ast.Constant(value=None))
            return

        self.stack.copy_from_top(depth)

    def swap_stack_item(self, opcode_depth: int) -> None:
        if opcode_depth < 1:
            self.warnings.append(f"SWAP with invalid depth: {opcode_depth}")
            return

        depth = opcode_depth - 1
        if len(self.stack) <= depth:
            self.warnings.append(f"SWAP_{opcode_depth} on shallow stack")
            return

        self.stack.swap_top(depth)

    def rotate_stack(self, count: int) -> None:
        if count < 2:
            return

        if len(self.stack) < count:
            self.warnings.append(f"ROT_{count} on shallow stack")
            return

        self.stack.rotate_top(count)

    def pop_binary(self) -> tuple[ast.expr, ast.expr]:
        right = self.expression_from_stack_value(self.pop_or_none())
        left = self.expression_from_stack_value(self.pop_or_none())
        return right, left

    def pop_many(self, count: int) -> list[ast.expr]:
        return [
            self.expression_from_stack_value(value)
            for value in self.pop_many_raw(count)
        ]

    def pop_many_raw(self, count: int) -> list[object]:
        if count < 0:
            raise ValueError("count must not be negative")
        if count <= len(self.stack):
            return self.stack.pop_many(count)

        values = [self.pop_or_none() for _index in range(count)]
        values.reverse()
        return values

    def pop_or_none(self) -> object:
        if self.stack:
            return self.stack.pop()

        warning = "bytecode stack underflow; inserted None placeholder"
        if warning not in self.warnings:
            self.warnings.append(warning)

        return ast.Constant(value=None)
