from langchain_core.runnables import RunnableLambda

# a chain in langchain is a RunnableSequence of Runnables, which can be thought of as a sequence of functions that are executed in order.
# The output of one Runnable is passed as input to the next Runnable in the sequence.
# The output of the last Runnable in the sequence is returned as the output of the chain.

from langchain_core.runnables import RunnableSequence


def add_one(x: int) -> int:
    return x + 1


def multi_two(x: int) -> int:
    return x * 2


runnable1 = RunnableLambda(add_one)
runnable2 = RunnableLambda(multi_two)

# or we can write as seq=RunnableSequnce(first=runnable1,last=runnable2) , both the first and last should have to be runnables
sequence = runnable1 | runnable2


res2 = sequence.invoke(5)
print(res2)

res = sequence.batch([1, 2, 3, 4, 5])
print(res)
