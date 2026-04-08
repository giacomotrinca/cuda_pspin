#!/bin/bash
# Build GCC 7.5.0 locally on kraken (openSUSE Leap 42)
# Installs to $HOME/gcc7  — no root needed
# Usage:  bash build_gcc7.sh

set -e

VER=7.5.0
PREFIX="$HOME/gcc7"
WORKDIR="$HOME/gcc7_build"

mkdir -p "$WORKDIR" "$PREFIX"
cd "$WORKDIR"

echo "=== Downloading GCC $VER ==="
wget --no-check-certificate https://ftp.gnu.org/gnu/gcc/gcc-${VER}/gcc-${VER}.tar.xz \
  || wget --no-check-certificate https://mirrors.kernel.org/gnu/gcc/gcc-${VER}/gcc-${VER}.tar.xz

echo "=== Extracting ==="
tar xf gcc-${VER}.tar.xz
cd gcc-${VER}

echo "=== Downloading prerequisites (gmp, mpfr, mpc, isl) ==="
# Use the bundled script but patch it to skip certificate checks
sed -i 's/wget/wget --no-check-certificate/g' contrib/download_prerequisites
./contrib/download_prerequisites

echo "=== Configuring ==="
mkdir -p build && cd build
../configure \
  --prefix="$PREFIX" \
  --enable-languages=c,c++ \
  --disable-multilib \
  --disable-bootstrap \
  --disable-nls \
  --with-default-libstdcxx-abi=new

echo "=== Building (this will take a while) ==="
NPROC=$(nproc 2>/dev/null || echo 4)
make -j"$NPROC"

echo "=== Installing to $PREFIX ==="
make install

echo ""
echo "=== Done! ==="
echo "GCC installed in: $PREFIX"
echo "Test:  $PREFIX/bin/g++ --version"
echo ""
echo "To use it, either:"
echo "  export PATH=$PREFIX/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=$PREFIX/lib64:\$LD_LIBRARY_PATH"
echo "or just use the full path in the Makefile."
