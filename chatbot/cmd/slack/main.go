// Experiment with the Slack API
//
// Object types: https://api.slack.com/types
package main

import (
	"flag"
	"fmt"
	log "github.com/sirupsen/logrus"
	"github.com/slack-go/slack"
	"os"
)

type options struct {
	Channel string
}
func main() {
	o := &options{}
	flag.StringVar(&o.Channel, "channel", "kubflowbot-playground", "Which channel to get messages for.")

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
		if channel.Name == o.Channel {
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


	// GetChannel history won't return replies to a thread
	log.Infof("Get channel messages")
	history,  err := api.GetChannelHistory(channelId, slack.HistoryParameters{
		Count: 10,
	})

	if err != nil {
		fmt.Printf("%s\n", err)
		return
	}

	for _, m := range history.Messages {
		// See: https://api.slack.com/messaging/retrieving#finding_threads#finding_threads
		// ThreadTimestamp and Timestamp are the same to indicate the start of a thread.
		if m.Timestamp == m.ThreadTimestamp {
			fmt.Println("Parent message")

			parameters := &slack.GetConversationRepliesParameters {
				ChannelID: channelId,
				Timestamp: m.ThreadTimestamp,
			}
			replies, _, _, err := api.GetConversationReplies(parameters)

			if err != nil {
				log.Errorf("Could not retrieve conversation replies; error: %v", err)
				continue
			}

			for i, r  := range replies {
				line := ""
				if i > 0 {
					line = line + "\t"
				}
				line = line + r.Text + "\n"
				fmt.Printf(line)
			}
		} else {
			fmt.Println(m.Text)
		}
		// channel is of type conversation & groupConversation
		// see all available methods in `conversation.go`
	}



}
