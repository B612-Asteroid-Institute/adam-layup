# adam-layup

A Python wrapper for Layup orbit determination software, designed to work seamlessly with adam_core.

Before first time call to orbit determination, call `LayupOrbitFitter.bootstrap()` to download necessary
data files. The bootstrap method takes a while the first time, but once the cache is filled it returns
very quickly.

Layup sometimes take a long time to process inputs. To work around that, the wrapper provides parameters
for subsetting the input observations and imposing timeout on each individual Layup run.