#pragma once
class Initialisation
{
public:
	Initialisation();
	~Initialisation();
private:
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanup();
};

//TODO: Migrate all of the initialisation functions and data to here

