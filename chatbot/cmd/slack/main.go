// Experiment with the Slack API
package main

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"github.com/slack-go/slack"
	"os"
)

func main() {
	// If you set debugging, it will log all requests to the console
	// Useful when encountering issues
	token := os.Getenv("SLACK_TOKEN")

	if token == "" {
		log.Fatal("You must set the environment variable SLACK_TOKEN")
	}

	api := slack.New(token, slack.OptionDebug(true))


	channels, err := api.GetChannels(false)
	if err != nil {
		fmt.Printf("%s\n", err)
		return
	}

	channelId := ""

	fmt.Println("Channel:\tIs Member:\tLastMessage:\t")
	for _, channel := range channels {
		if channel.Name == "general" {
			channelId = channel.ID
		}

		last := ""
		if channel.Latest != nil {
			last = channel.Latest.Text
		}
		fmt.Printf("%v\t%v\t%v\n", channel.Name, channel.IsMember, last)
		// channel is of type conversation & groupConversation
		// see all available methods in `conversation.go`
	}


	log.Infof("Get channel messages")
	history,  err := api.GetChannelHistory(channelId, slack.HistoryParameters{})

	if err != nil {
		fmt.Printf("%s\n", err)
		return
	}

	for _, m := range history.Messages {
		fmt.Println(m.Text)
		// channel is of type conversation & groupConversation
		// see all available methods in `conversation.go`
	}


	//parameters := &slack.GetConversationRepliesParameters {
	//	ChannelID: targetChannel.ID,
	//}
	//replies, _, _, err := api.GetConversationReplies(parameters)
}
