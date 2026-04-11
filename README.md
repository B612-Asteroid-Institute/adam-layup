# adam-layup

A Python wrapper for Layup orbit determination software, designed to work seamlessly with adam_core.

Before first time call to orbit determination, call `LayupOrbitFitter.bootstrap()` to download necessary
data files. The bootstrap method takes a while the first time, but once the cache is filled it returns
very quickly.