# Sourced by the test scripts, after CUDA_VISIBLE_DEVICES is set.
#
# What the proxy keeps of the modules a program loads, so that the next start neither adapts nor
# translates them again (VF_PROXY_CACHE), goes by default to the home directory of whoever runs:
# on the nodes that is memory, and up to 2 GB of it. It cannot go to the NFS either: the proxy
# refuses a cache whose files do not belong to the user it runs as, and root's files there do not.
# Each run gets a directory under $TMPDIR (/tmp when it is not set), which must be a local directory
# of whoever runs, and removes it on exit. Scripts with their own exit trap call
# vf_cleanup_proxy_cache. A caller can supply VF_PROXY_CACHE, or 0 to disable it.
_vf_owned_proxy_cache=
vf_cleanup_proxy_cache() {
    if [ -n "$_vf_owned_proxy_cache" ]; then rm -rf -- "$_vf_owned_proxy_cache"; fi
}
if [ -z "${VF_PROXY_CACHE:-}" ]; then
    _vf_owned_proxy_cache=$(mktemp -d "${TMPDIR:-/tmp}/vf-cache-XXXXXXXX") || exit 1
    export VF_PROXY_CACHE=$_vf_owned_proxy_cache
    trap vf_cleanup_proxy_cache EXIT
fi

# Two more things a program writes under the home directory of whoever runs it, which on the nodes
# is memory: the kernels the driver compiles from PTX (~/.nv/ComputeCache: 212 MB were found there
# on 2026-11-19, from tests that did not say otherwise) and the kernels PyTorch compiles by itself
# (~/.cache/torch/kernels). With the other outputs instead, unless the test or the caller has said
# where. (Not llama.cpp's directory of compiled kernels: a test with the driver's default limit of
# 1 GB would empty what took minutes to compile.)
_vf_out=${VF_OUT:-$VF_TEST_ROOT}
if [ -d "$_vf_out" ]; then
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$_vf_out/nv-cache}
    export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-$_vf_out/torch-kernels}
fi
unset _vf_out
