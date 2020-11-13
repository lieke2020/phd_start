{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "\n",
    "def TicTocGenerator():\n",
    "    # Generator that returns time differences\n",
    "    ti = 0           # initial time\n",
    "    tf = time.time() # final time\n",
    "    while True:\n",
    "        ti = tf\n",
    "        tf = time.time()\n",
    "        yield tf-ti # returns the time difference\n",
    "\n",
    "TicToc = TicTocGenerator() # create an instance of the TicTocGen generator\n",
    "\n",
    "# This will be the main function through which we define both tic() and toc()\n",
    "def toc(tempBool=True):\n",
    "    # Prints the time difference yielded by generator instance TicToc\n",
    "    tempTimeInterval = next(TicToc)\n",
    "    if tempBool:\n",
    "        print( \"Elapsed time: %f seconds.\\n\" %tempTimeInterval )\n",
    "\n",
    "def tic():\n",
    "    # Records a time in TicToc, marks the beginning of a time interval\n",
    "    toc(False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
