/// @file
///
/// This example shows how to compile and run drake files

#include <gflags/gflags.h>

#include "drake/common/text_logging_gflags.h"

namespace drake {
namespace examples {
namespace hello {

DEFINE_string(your_name, "Zion",
              "Putting your name here so Drake recognize you.");

void DoMain() {
  drake::log()->info("Hello " + FLAGS_your_name + " from Drake!");
}

}  // namespace hello
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple hello Drake example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::hello::DoMain();
  return 0;
}
