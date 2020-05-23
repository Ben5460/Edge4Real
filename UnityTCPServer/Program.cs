using System.Threading;
using System;

namespace TCPServer
{
    class Program
    {
        static void Main(string[] args)
        {
            Thread serverThread = new Thread(() => new Server("192.168.0.47" , 8080));
            serverThread.Start();
            
            // server is going to be launch from here
            Console.WriteLine("Server Started...192.168.0.47");
        }
    }
}
