#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Exp10LanSimulation");

int main (int argc, char *argv[])
{
  LogComponentEnable ("Exp10LanSimulation", LOG_LEVEL_INFO);

  // 1. Create a container of 3 nodes
  NodeContainer nodes;
  nodes.Create (3);

  // 2. Configure CSMA channel and device helper
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));

  // 3. Install network devices on nodes
  NetDeviceContainer devices;
  devices = csma.Install (nodes);

  // 4. Install Internet Stack
  InternetStackHelper stack;
  stack.Install (nodes);

  // 5. Assign IP addresses in 10.1.1.0/24 subnet
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  // 6. Print assigned IP addresses to stdout
  for (uint32_t i = 0; i < nodes.GetN (); i++)
    {
      std::cout << "Node " << i << " IP Address: " 
                << interfaces.GetAddress (i) << std::endl;
    }

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
