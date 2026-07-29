//===- HotswapHsaToolDispatchTest.cpp ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-dispatch.h"
#include "hotswap-platform-io.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <sys/types.h>
#include <unistd.h>

using namespace COMGR::hotswap::hsa_tool;

namespace {

TEST(HotswapHsaToolDispatch, RewritesRegisteredKernelAndScalesX) {
  KernelDispatchTarget Target{/*KernelObject=*/0x1234000,
                              /*SourcePrivateSegmentSize=*/64,
                              /*TargetPrivateSegmentSize=*/96,
                              /*SourceGroupSegmentSize=*/128,
                              /*TargetGroupSegmentSize=*/160,
                              /*Scale=*/2,
                              /*MaxWorkgroupSizeX=*/1024,
                              /*MaxWorkgroupSizeY=*/1024,
                              /*MaxWorkgroupSizeZ=*/1024,
                              /*MaxWorkgroupSize=*/1024,
                              /*MaxGridSizeX=*/UINT32_MAX,
                              /*MaxGridSizeY=*/UINT32_MAX,
                              /*MaxGridSizeZ=*/UINT32_MAX,
                              /*MaxGridSize=*/UINT64_MAX};
  KernelDispatchPacket Packet{/*KernelObject=*/0xfeed,
                              /*PrivateSegmentSize=*/96,
                              /*GroupSegmentSize=*/256,
                              /*WorkgroupSizeX=*/32,
                              /*WorkgroupSizeY=*/1,
                              /*WorkgroupSizeZ=*/1,
                              DispatchGrid{/*X=*/1024, /*Y=*/1, /*Z=*/1}};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet), DispatchRewriteError::None);
  EXPECT_EQ(Packet.KernelObject, Target.KernelObject);
  EXPECT_EQ(Packet.PrivateSegmentSize, 128u);
  EXPECT_EQ(Packet.GroupSegmentSize, 288u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 64u);
  ASSERT_TRUE(Packet.Grid);
  EXPECT_EQ(Packet.Grid->X, 2048u);
}

TEST(HotswapHsaToolDispatch, RejectsMissingPhysicalTarget) {
  KernelDispatchTarget Target;
  KernelDispatchPacket Packet{/*KernelObject=*/1,
                              /*PrivateSegmentSize=*/0,
                              /*GroupSegmentSize=*/0,
                              /*WorkgroupSizeX=*/1,
                              /*WorkgroupSizeY=*/1,
                              /*WorkgroupSizeZ=*/1,
                              DispatchGrid{/*X=*/1, /*Y=*/1, /*Z=*/1}};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::InvalidTarget);
  EXPECT_EQ(Packet.KernelObject, 1u);
}

TEST(HotswapHsaToolDispatch, RejectsInsufficientSegments) {
  KernelDispatchTarget Target{/*KernelObject=*/2,
                              /*SourcePrivateSegmentSize=*/64,
                              /*TargetPrivateSegmentSize=*/32,
                              /*SourceGroupSegmentSize=*/128,
                              /*TargetGroupSegmentSize=*/64,
                              /*Scale=*/1,
                              /*MaxWorkgroupSizeX=*/1024,
                              /*MaxWorkgroupSizeY=*/1024,
                              /*MaxWorkgroupSizeZ=*/1024,
                              /*MaxWorkgroupSize=*/1024,
                              /*MaxGridSizeX=*/UINT32_MAX,
                              /*MaxGridSizeY=*/UINT32_MAX,
                              /*MaxGridSizeZ=*/UINT32_MAX,
                              /*MaxGridSize=*/UINT64_MAX};
  KernelDispatchPacket Packet{/*KernelObject=*/1,
                              /*PrivateSegmentSize=*/63,
                              /*GroupSegmentSize=*/128,
                              /*WorkgroupSizeX=*/1,
                              /*WorkgroupSizeY=*/1,
                              /*WorkgroupSizeZ=*/1,
                              /*Grid=*/std::nullopt};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::PrivateSegmentTooSmall);
  Packet.PrivateSegmentSize = 64;
  Packet.GroupSegmentSize = 127;
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::GroupSegmentTooSmall);
}

TEST(HotswapHsaToolDispatch, RejectsScaleOverflowWithoutPartialRewrite) {
  KernelDispatchTarget Target{/*KernelObject=*/2,
                              /*SourcePrivateSegmentSize=*/0,
                              /*TargetPrivateSegmentSize=*/0,
                              /*SourceGroupSegmentSize=*/0,
                              /*TargetGroupSegmentSize=*/0,
                              /*Scale=*/2,
                              /*MaxWorkgroupSizeX=*/1024,
                              /*MaxWorkgroupSizeY=*/1024,
                              /*MaxWorkgroupSizeZ=*/1024,
                              /*MaxWorkgroupSize=*/1024,
                              /*MaxGridSizeX=*/UINT32_MAX,
                              /*MaxGridSizeY=*/UINT32_MAX,
                              /*MaxGridSizeZ=*/UINT32_MAX,
                              /*MaxGridSize=*/UINT64_MAX};
  KernelDispatchPacket Packet{
      /*KernelObject=*/1,
      /*PrivateSegmentSize=*/0,
      /*GroupSegmentSize=*/0,
      /*WorkgroupSizeX=*/std::numeric_limits<uint16_t>::max(),
      /*WorkgroupSizeY=*/1,
      /*WorkgroupSizeZ=*/1,
      DispatchGrid{/*X=*/1, /*Y=*/1, /*Z=*/1}};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::WorkgroupSizeOverflow);
  EXPECT_EQ(Packet.KernelObject, 1u);
  ASSERT_TRUE(Packet.Grid);
  EXPECT_EQ(Packet.Grid->X, 1u);

  Packet.WorkgroupSizeX = 1;
  Packet.Grid->X = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::GridSizeOverflow);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 1u);
}

TEST(HotswapHsaToolDispatch, RejectsScaledWorkgroupsBeyondPhysicalLimits) {
  KernelDispatchTarget Target{/*KernelObject=*/2,
                              /*SourcePrivateSegmentSize=*/0,
                              /*TargetPrivateSegmentSize=*/0,
                              /*SourceGroupSegmentSize=*/0,
                              /*TargetGroupSegmentSize=*/0,
                              /*Scale=*/2,
                              /*MaxWorkgroupSizeX=*/1024,
                              /*MaxWorkgroupSizeY=*/1024,
                              /*MaxWorkgroupSizeZ=*/1024,
                              /*MaxWorkgroupSize=*/1024,
                              /*MaxGridSizeX=*/UINT32_MAX,
                              /*MaxGridSizeY=*/UINT32_MAX,
                              /*MaxGridSizeZ=*/UINT32_MAX,
                              /*MaxGridSize=*/UINT64_MAX};
  KernelDispatchPacket Packet{/*KernelObject=*/1,
                              /*PrivateSegmentSize=*/0,
                              /*GroupSegmentSize=*/0,
                              /*WorkgroupSizeX=*/1024,
                              /*WorkgroupSizeY=*/1,
                              /*WorkgroupSizeZ=*/1,
                              DispatchGrid{/*X=*/1024, /*Y=*/1, /*Z=*/1}};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::WorkgroupDimensionExceeded);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 1024u);
  ASSERT_TRUE(Packet.Grid);
  EXPECT_EQ(Packet.Grid->X, 1024u);

  Packet.WorkgroupSizeX = 256;
  Packet.WorkgroupSizeY = 4;
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::WorkgroupSizeExceeded);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 256u);

  Packet.WorkgroupSizeX = 1;
  Packet.WorkgroupSizeY = 3;
  Target.MaxWorkgroupSizeY = 2;
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::WorkgroupDimensionExceeded);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 1u);
}

TEST(HotswapHsaToolDispatch,
     RejectsScaledGridBeyondPhysicalLimitsWithoutPartialRewrite) {
  KernelDispatchTarget Target{/*KernelObject=*/2,
                              /*SourcePrivateSegmentSize=*/0,
                              /*TargetPrivateSegmentSize=*/0,
                              /*SourceGroupSegmentSize=*/0,
                              /*TargetGroupSegmentSize=*/0,
                              /*Scale=*/2,
                              /*MaxWorkgroupSizeX=*/1024,
                              /*MaxWorkgroupSizeY=*/1024,
                              /*MaxWorkgroupSizeZ=*/1024,
                              /*MaxWorkgroupSize=*/1024,
                              /*MaxGridSizeX=*/16,
                              /*MaxGridSizeY=*/16,
                              /*MaxGridSizeZ=*/16,
                              /*MaxGridSize=*/64};
  KernelDispatchPacket Packet{/*KernelObject=*/1,
                              /*PrivateSegmentSize=*/0,
                              /*GroupSegmentSize=*/0,
                              /*WorkgroupSizeX=*/1,
                              /*WorkgroupSizeY=*/1,
                              /*WorkgroupSizeZ=*/1,
                              DispatchGrid{/*X=*/9, /*Y=*/1, /*Z=*/1}};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::GridDimensionExceeded);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 1u);
  ASSERT_TRUE(Packet.Grid);
  EXPECT_EQ(Packet.Grid->X, 9u);

  Packet.Grid->X = 8;
  Packet.Grid->Y = 5;
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::GridSizeExceeded);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.WorkgroupSizeX, 1u);
  EXPECT_EQ(Packet.Grid->X, 8u);
}

TEST(HotswapHsaToolDispatch, RejectsSegmentOverflowWithoutPartialRewrite) {
  KernelDispatchTarget Target{/*KernelObject=*/2,
                              /*SourcePrivateSegmentSize=*/0,
                              /*TargetPrivateSegmentSize=*/1,
                              /*SourceGroupSegmentSize=*/0,
                              /*TargetGroupSegmentSize=*/0,
                              /*Scale=*/1,
                              /*MaxWorkgroupSizeX=*/1024,
                              /*MaxWorkgroupSizeY=*/1024,
                              /*MaxWorkgroupSizeZ=*/1024,
                              /*MaxWorkgroupSize=*/1024,
                              /*MaxGridSizeX=*/UINT32_MAX,
                              /*MaxGridSizeY=*/UINT32_MAX,
                              /*MaxGridSizeZ=*/UINT32_MAX,
                              /*MaxGridSize=*/UINT64_MAX};
  KernelDispatchPacket Packet{
      /*KernelObject=*/1,
      /*PrivateSegmentSize=*/std::numeric_limits<uint32_t>::max(),
      /*GroupSegmentSize=*/7,
      /*WorkgroupSizeX=*/1,
      /*WorkgroupSizeY=*/1,
      /*WorkgroupSizeZ=*/1,
      /*Grid=*/std::nullopt};
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::PrivateSegmentOverflow);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.PrivateSegmentSize, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(Packet.GroupSegmentSize, 7u);

  Target.TargetPrivateSegmentSize = 0;
  Target.TargetGroupSegmentSize = 1;
  Packet.PrivateSegmentSize = 7;
  Packet.GroupSegmentSize = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(rewriteKernelDispatch(Target, Packet),
            DispatchRewriteError::GroupSegmentOverflow);
  EXPECT_EQ(Packet.KernelObject, 1u);
  EXPECT_EQ(Packet.PrivateSegmentSize, 7u);
  EXPECT_EQ(Packet.GroupSegmentSize, std::numeric_limits<uint32_t>::max());
}

TEST(HotswapHsaToolPlatformIO, ReadsFileSliceWithoutChangingFileOffset) {
  char Path[] = "/tmp/comgr-hotswap-hsa-tool-XXXXXX";
  const int File = mkstemp(Path);
  ASSERT_NE(File, -1);
  ASSERT_EQ(unlink(Path), 0);
  constexpr char Contents[] = "0123456789";
  ASSERT_EQ(pwrite(File, Contents, sizeof(Contents) - 1, 0),
            static_cast<ssize_t>(sizeof(Contents) - 1));
  ASSERT_EQ(lseek(File, 7, SEEK_SET), 7);

  const Bytes Slice = readFile(File, 2, 4);
  ASSERT_TRUE(Slice);
  EXPECT_EQ(std::string(Slice->begin(), Slice->end()), "2345");
  EXPECT_EQ(lseek(File, 0, SEEK_CUR), 7);

  const Bytes Whole = readWholeFile(File);
  ASSERT_TRUE(Whole);
  EXPECT_EQ(std::string(Whole->begin(), Whole->end()), "0123456789");
  EXPECT_EQ(lseek(File, 0, SEEK_CUR), 7);
  EXPECT_EQ(close(File), 0);
}

TEST(HotswapHsaToolPlatformIO, RejectsInvalidRangesAndShortReads) {
  char Path[] = "/tmp/comgr-hotswap-hsa-tool-XXXXXX";
  const int File = mkstemp(Path);
  ASSERT_NE(File, -1);
  ASSERT_EQ(unlink(Path), 0);
  constexpr char Contents[] = "abc";
  ASSERT_EQ(pwrite(File, Contents, sizeof(Contents) - 1, 0),
            static_cast<ssize_t>(sizeof(Contents) - 1));

  EXPECT_FALSE(readFile(File, 0, 0));
  EXPECT_FALSE(readFile(File, 2, 2));
  EXPECT_FALSE(readFile(File, std::numeric_limits<uint64_t>::max(), 1));
  EXPECT_EQ(close(File), 0);
}

} // namespace
